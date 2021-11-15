import sys
from math import log2, ceil
from inspect import isfunction
from collections import namedtuple

# import torch
# import torch.utils.checkpoint
# from torch import nn

import torch
from torch import nn, einsum, diagonal
import torch.nn.functional as F
from torch.autograd import Function

from einops import rearrange, repeat

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.utils import logging
from .configuration_htransformer1d import HTransformer1DConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HTransformer1DConfig"
_TOKENIZER_FOR_DOC = "HTransformer1DTokenizer"

# Define named tuples for nn.Modules here
AttentionOutput = namedtuple("AttentionOutput", ["hidden_states", "attentions"])
HTransformer1DLayerOutput = namedtuple(
    "HTransformer1DLayerOutput", 
    ["hidden_states", "attn_output", "attentions"]
)
HTransformer1DBackwardOutput = namedtuple(
    "HTransformer1DBackwardOutput",
    ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"]
)


# hierarchical attention helper functions

def flip_every_two(t):
    t = rearrange(t, 'b (n r) ... -> b n r ...', r = 2)
    t = torch.flip(t, dims = (2,))                          # so we pay attention to the off-diagonal blocks in the attention matrix
    t = rearrange(t, 'b n r ... -> b (n r) ...')
    return t


# lucidrains/rotary-embedding-torch
# ./rotary_embedding_torch/rotary_embedding_torch.py#L60
class HTransformer1DRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        theta = config.rotary_theta
        dim = config.dim_head
        freqs = 1. / (theta ** torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)
        self.cache = dict()
        
        if config.learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)
    
    def forward(self, t, cache_key=None):
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]
        
        if isfunction(t):
            t = t()
            
        freqs = self.freqs
        
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        
        if cache_key is not None:
            self.cache[cache_key] = freqs
            
        return freqs
    
    def extra_repr(self):
        extra_repr = (
            f"dim={self.config.dim_head}, "
            f"theta={self.config.rotary_theta}, "
            f"learned_freq={self.config.learned_freq}"
        )
        return extra_repr
    
    @staticmethod
    def apply_rotary_emb(freqs, t, start_index=0):
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], (
            f'feature dimension {t.shape[-1]} is not of sufficient '
            f'size to rotate in all the positions {rot_dim}'
        )
        t_left = t[..., :start_index]
        t = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        
        def rotary_half(x):
            x = rearrange(x, '... (d r) -> ... d r', r = 2)
            x1, x2 = x.unbind(dim = -1)
            x = torch.stack((-x2, x1), dim = -1)
            return rearrange(x, '... d r -> ... (d r)')
        
        t = (t * freqs.cos()) + (rotary_half(t) * freqs.sin())
        return torch.cat((t_left, t, t_right), dim=-1)


class HTransformer1DEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = None
        self.padding_idx = config.pad_token_id
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == "absolute":
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self, 
        input_ids=None, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
                
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            
        return embeddings
    
    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.
        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(self.padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + self.padding_idx
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# lucidrains/h-transformer-1d
# h_transformer_1d/h_transformer_1d.py#L100
class HTransformer1DAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.heads = config.num_attention_heads
        self.dim_head = config.dim_head
        self.eps = config.attn_eps
        self.scale = self.dim_head ** -0.5
        self.position_embedding_type = config.position_embedding_type
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        inner_dim = self.heads * self.dim_head
        
        self.to_qkv = nn.Linear(config.hidden_size, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, config.hidden_size)
        
        if self.position_embedding_type == "rotary":
            # Whether or not to apply rotary to value
            self.rotary_value = config.rotary_value
            self.pos_emb = HTransformer1DRotaryEmbedding(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        # pad sequence length to power of 2
        if seq_length >= self.block_size * 2:
            pad_to_len = 2 ** ceil(log2(seq_length))
        else:
            pad_to_len = self.block_size * 2
        padding = pad_to_len - seq_length
        
        # PreLN architectures
        hidden_states = self.layer_norm(hidden_states)
        
        if padding != 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding), value=0.)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, padding), value=False)
        
        # derive queries, keys, values
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        
        # split out heads, and also divide sequence into blocks
        to_heads = lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads)
        q, k, v = map(to_heads, (q, k, v))
        
        if attention_mask is not None:
            attention_mask = repeat(attention_mask, 'b n -> (b h) n', h=self.heads)
            
        # scale
        q = q * self.scale
        
        # rotary pos emb
        if self.position_embedding_type == "rotary":
            add_to_pad = torch.arange(pad_to_len, device=q.device)
            freqs = self.pos_emb(add_to_pad, cache_key=pad_to_len)
            freqs = rearrange(freqs, 'n d -> () n d')
            apply_rotary_emb = self.pos_emb.apply_rotary_emb
            q, k = map(lambda t: apply_rotary_emb(freqs, t), (q, k))
            if self.rotary_value:
                v = apply_rotary_emb(freqs, v)
            
        # calculate number of levels until 2 x 2
        num_levels = int(log2(pad_to_len // self.block_size)) - 1
        assert num_levels >= 0, 'number of levels must be at least greater than 0'
        
        # coarsening
        qkvs = [(q, k, v, attention_mask)]
        for level in range(num_levels):
            coarsened_qkvs = self.coarsen(*qkvs[-1])
            qkvs.append(coarsened_qkvs)
        qkvs = [qkvs[0], *qkvs] # duplicate the finest resolution an extra time, for the base diagonal
        
        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = self.block_size)
        # calculate Ys, ad in the paper
        Ys = []
        for ind, (q, k, v, mask) in enumerate(reversed(qkvs)):
            is_last = ind == (len(qkvs) - 1)            
            q, k, v = map(to_blocks, (q, k, v))
            
            # generate the mask for S
            S_mask = None
            if mask is not None:
                mask = to_blocks(mask)
                q_mask = mask
                k_mask = flip_every_two(mask) if not is_last else mask
                S_mask = rearrange(q_mask, '... n -> ... n ()') * rearrange(k_mask, '... n -> ... () n')
                
            # flip keys and values to capture the off-diagonal
            if not is_last:
                k, v = map(flip_every_two, (k, v))
                
            y, A = self.compute_Y_and_A(q, k, v, mask=S_mask)
            Ys.append((y, A))
            
        # interpolate
        Y, A, attentions = 0, 0, 0
        for ind, (Y_level, A_level) in enumerate(Ys):
            is_last = ind == (len(Ys) - 1)
            
            if not is_last and torch.is_tensor(Y):
                Y = repeat(Y, 'b n d -> b (n r) d', r=2)
                
            if not is_last and torch.is_tensor(A):
                A = repeat(A, 'b n -> b (n r)', r=2)
                
            Y = Y_level + Y
            A = A_level + A
            
        out = Y / rearrange(A + self.eps, 'b n -> b n ()')
        # merge heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        # combine out
        hidden_states = self.to_out(out[:, :seq_length])
    
        return AttentionOutput(
            hidden_states=hidden_states,
            attentions=None, # Not implemented
        )
    
    @staticmethod
    def coarsen(q, k, v, mask=None):
        to_coarse = lambda t: rearrange(t, 'b (n r) d -> b n r d', r=2)
        q, k, v = map(to_coarse, (q, k, v))
        if mask is not None:
            mask = repeat(mask, 'b (n r) -> b n r', r=2)
            
        # masked mean for queries and keys, but not values
        def masked_aggregate(tensor, mask=None, dim=-1, average=True):
            if mask is None:
                fn = torch.sum if not average else torch.mean
                return fn(tensor, dim=dim)
            
            diff_len = len(tensor.shape) - len(mask.shape)
            mask = mask[(..., *((None,) * diff_len))]
            tensor = tensor.masked_fill(~mask, 0.)
            
            total_el = mask.sum(dim=dim)
            agg = tensor.sum(dim=dim)
            
            if average:
                agg = agg / total_el.clamp(min=1.)
                
            agg.masked_fill_(total_el == 0, 0.)
            return agg
        
        q = masked_aggregate(q, mask, dim=2)
        k = masked_aggregate(k, mask, dim=2)
        v = masked_aggregate(v, mask, dim=2, average=False)
        
        if mask is not None:
            mask = torch.any(mask, dim=2)
            
        return (q, k, v, mask)
    
    @staticmethod
    def compute_Y_and_A(q, k, v, mask=None):
        S = einsum('... i d, ... j d -> ... i j', q, k)
        
        if mask is not None:
            mask_value = -torch.finfo(S.dtype).max
            S = S.masked_fill(~mask, mask_value)
            
        S = S - torch.max(S, dim=-1, keepdim=True).values
        A = S.exp()
        
        y = einsum('... i j, ... j d -> ... i d', A, v)
        
        A = A.sum(dim=-1)
        
        y = rearrange(y, 'b ... n d -> b (... n) d')
        A = rearrange(A, 'b ... i -> b (... i)')
        
        return y, A


class HTransformer1DIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
            
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dropout(self.dense(hidden_states))
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class HTransformer1DOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dropout(self.dense(hidden_states))
        return hidden_states
    

# Copied from transformers.models.reformer.modeling_reformer.ChunkReformerFeedForward with Reformer->HTransformer1D
class ChunkHTransformer1DFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = HTransformer1DIntermediate(config)
        self.output = HTransformer1DOutput(config)

    def forward(self, attention_output, **kwargs):
        return apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

    def forward_chunk(self, hidden_states):
        # PreLN architectures
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.output(hidden_states)
        return hidden_states
    

# lucidrains/h-transformer-1d
# ./h_transformer_1d/h_transformer_1d.py#L73
class PreShiftToken(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.fn = module(config)
        self.shifts = (0, 1) if config.shift_tokens else (-1, 0, 1)
        self.config = config
        
    def _shift(self, t, amount, mask=None):
        if amount == 0:
            return t

        if mask is not None:
            t = t.masked_fill(~mask[..., None], 0.)

        return F.pad(t, (0, 0, amount, -amount), value=0.)
    
    def forward(self, hidden_states, **kwargs):
        if self.config.shift_tokens:
            mask = kwargs.get('attention_mask', None)
            shifts = self.shifts
            segments = len(shifts)
            feats_per_shift = hidden_states.shape[-1] // segments
            splitted = hidden_states.split(feats_per_shift, dim=-1)
            segments_to_shift, rest = splitted[:segments], splitted[segments:]
            segments_to_shift = list(
                map(lambda args: self._shift(*args, mask=mask), 
                    zip(segments_to_shift, shifts))
            )
            hidden_states = torch.cat((*segments_to_shift, *rest), dim=-1)
            kwargs["attention_mask"] = mask
        return self.fn(hidden_states, **kwargs)
    
    def __repr__(self):
        fn_repr = self.fn.__repr__()
        if self.config.shift_tokens:
            cls_name = self.__class__.__name__
            fn_repr = cls_name + " " + fn_repr
        return fn_repr
    
    
class HTransformer1DSequentialLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PreShiftToken(HTransformer1DAttention, config)
        self.feed_forward  = PreShiftToken(ChunkHTransformer1DFeedForward, config)
    
    def forward(
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        # x <- x + f(x)
        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs.hidden_states + hidden_states
        
        # x <- x + g(x)
        layer_outputs = self.feed_forward(attn_output)
        hidden_states = attn_output + layer_outputs
        
        return HTransformer1DLayerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attentions=attn_outputs.attentions
        )

    
class HTransformer1DReversibleLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PreShiftToken(HTransformer1DAttention, config)
        self.feed_forward  = PreShiftToken(ChunkHTransformer1DFeedForward, config)
        self.attention_seed = None
        self.feed_forward_seed = None
        
    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        # randomize seeds
        # use cuda generator if available
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.attention_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
        # randomize seeds
        # use cuda generator if available
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.feed_forward_seed)
    
    def forward(
        self,
        prev_attn_output,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # X_1: prev_attn_output
        # X_2: hidden_states
        with torch.no_grad():
            # every forward pass we sample a different seed
            # for dropout and save for forward fn in backward pass
            # to have correct dropout
            if self.training:
                self._init_attention_seed()
                
            attn_outputs = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs.hidden_states
            
            # Implementation of RevNet
            # (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
            # Y_1 = X_1 + f(X_2)
            attn_output = prev_attn_output + attn_output
            
            # free memory
            del prev_attn_output
            
            # every forward pass we sample a different seed
            # for dropout and save for forward fn in backward
            # to have correct dropout
            if self.training:
                self._init_feed_forward_seed()
            # Y_2 = X_2 + g(Y_1)
            hidden_states = hidden_states + self.feed_forward(attn_output)
            
        return HTransformer1DLayerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attentions=attn_outputs.attentions
        )
    
    def backward_pass(
        self,
        next_attn_output,
        hidden_states,
        grad_attn_output,
        grad_hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        # Implements the backward pass for reversible block
        # A good blog post on how this works can be found here:
        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # This code is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
        
        assert self.training, (
            "If you want to train `HTransformer1DModel` and its variations, make sure "
            "to use `model.train()` to put the model into training mode."
        )
        
        # Y_1: next_attn_output
        # Y_2: hidden_states
        # dY_1: grad_attn_output
        # dY_2: grad_hidden_states
        with torch.enable_grad():
            next_attn_output.requires_grad = True
            # set seed to have correct dropout
            torch.manual_seed(self.feed_forward_seed)
            # g(Y_1) where g: feed forward module
            res_hidden_states = self.feed_forward(nex_attn_output)            
            # calculate gradient: g(Y_1) / dY_2
            res_hidden_states.backward(grad_hidden_states, retain_graph=True)
            
        with torch.no_grad():
            # X_2 = Y_2 - g(Y_1)
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states
            # dX_1 = dY_1 + X_2.grad
            grad_attn_output = grad_attn_output + next_attn_output.grad
            next_attn_output.grad = None # X_2.grad = None
            
        with torch.enable_grad():
            hidden_states.requires_grad = True # X_2.grad = True
            # set seed to have correct dropout
            torch.manual_seed(self.attention_seed)
            # f(X_2) where f: self-attention module
            output = self.attention(
                hidden_states=hidden_states,
                head_mask=head_mask,
                attention_mask=attention_mask,
            ).hidden_states
            # calculate gradient: f(X_2) / dX_1
            output.backward(grad_attn_output, retain_graph=True)
            
        with torch.no_grad():
            # X_1 = Y_1 - f(X_2)
            attn_output = next_attn_output - output
            del output, next_attn_output
            # dX_2 = dY_2 + X_2.grad
            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.grad = None # X_2.grad = None
            hidden_states = hidden_states.detach()
            
        return HTransformer1DBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )


class HTransformer1DSequentialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            self.layers.append(HTransformer1DSequentialLayer(config))
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer, layer_head_mask in zip(self.layers, head_mask):
            if output_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            
            if output_attentions:
                all_attentions.append(layer_outputs.attentions)
                
        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class _ReversibleFunction(Function):
    """
    To prevent PyTorch from performing the usual backpropagatio, a customized backward function is implemented here.
    This way it is made sure that no memory expensive activation are saved during the forward pass. This function is
    heavily inspired by https://github.com/lucidrains/h-transformer-1d/blob/main/h_transformer_1d/reversible.py
    """
    
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        all_hidden_states,
        all_attentions,
        output_hidden_states,
        output_attentions
    ):
        # split duplicated tensor
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)
        
        for layer, layer_head_mask in zip(layers, head_mask):
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)
            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            
            if output_attentions:
                all_attentions.append(layer_outputs.attentions)
                
        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)
        
        # attach params to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask
        
        # Concatenate 2 RevNet outputs
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        # dy: grad_hidden_states
        # dY_1: grad_attn_output
        # dY_2: grad_hidden_states
        grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)
        
        # retrieve params from ctx for backward
        # X_1: attn_output
        # X_2: hidden_states
        attn_output, hidden_states = ctx.saved_tensors
        
        # create tuple
        output = HTransformer1DBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )
        
        # free memory
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states
        
        layers = ctx.layers
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask
        
        for idx, layer in enumerate(layers[::-1]):
            # backprop
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hidden_states=output.hidden_states,
                grad_attn_output=output.grad_attn_output,
                grad_hidden_states=output.grad_hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[len(layers)-idx-1],
            )
            
        grad_hidden_states = torch.cat(
            [output.grad_attn_output, output.grad_hidden_states], dim=-1)
        
        # num of return vars has to match num of forward() args
        # return gradient for hidden_states arg and None of other args
        return grad_hidden_states, None, None, None, None, None


class HTransformer1DReversibleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            self.layers.append(HTransformer1DReversibleLayer(config))
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # concat same tensor for reversible ResNet [X1, X2]
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            all_hidden_states,
            all_attentions,
            output_hidden_states,
            output_attentions,
        )
        # attn_output + hidden_states
        hidden_states = torch.stack(hidden_states.chunk(2, dim=-1)).sum(dim=0)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )        


# class HTransformer1DPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.embedding_size)
#         if isinstance(config.hidden_act, str):
#             self.transform_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.transform_act_fn = config.hidden_act
#         self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states
    
    
# class HTransformer1DLMPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.transform = HTransformer1DPredictionHeadTransform(config)

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))

#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias

#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states


# # Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->HTransformer1D
# class HTransformer1DOnlyMLMHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.predictions = HTransformer1DLMPredictionHead(config)

#     def forward(self, sequence_output):
#         prediction_scores = self.predictions(sequence_output)
#         return prediction_scores


class HTransformer1DPreTrainedModel(PreTrainedModel):
    config_class = HTransformer1DConfig
    base_model_prefix = "h_transformer_1d"
    # supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, HTransformer1DRotaryEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class HTransformer1DModel(HTransformer1DPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = HTransformer1DEmbeddings(config)
        if config.reversible:
            self.encoder = HTransformer1DReversibleEncoder(config)
        else:
            self.encoder = HTransformer1DSequentialEncoder(config)
        
        self.init_weights()
        
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        output_attentions=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        assert seq_length <= self.config.max_position_embeddings, (
            'sequence length must be less than the maximum sequence length'
        )
        
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        attention_mask = attention_mask.to(torch.bool)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attentions has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        embedding_output = self.embeddings(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds,
        )
        
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class HTransformer1DForMaskedLM(HTransformer1DPreTrainedModel):
    def __init__(self, config):
        pass
    
    def forward(self):
        pass


# class HTransformer1DForCausalLM(HTransformer1DPreTrainedModel):
#     def __init__(self, config):
#         pass
    
#     def forward(self):
#         pass


class HTransformer1DClassificationHead(nn.Module):
    def __init__(self, config):
        pass
    
    def forward(self):
        pass


class HTransformer1DForSequenceClassification(nn.Module):
    
    def __init__(self, config):
        pass
    
    def forward(self):
        pass


class HTransformer1DForMultipleChoice(HTransformer1DPreTrainedModel):
    def __init__(self, config):
        pass
    
    def forward(self):
        pass


class HTransformer1DForTokenClassification(HTransformer1DPreTrainedModel):
    def __init__(self, config):
        pass
    
    def forward(self):
        pass


class HTransformer1DForQuestionAnswering(HTransformer1DPreTrainedModel):
    def __init__(self, config):
        pass
    
    def forward(self):
        pass
