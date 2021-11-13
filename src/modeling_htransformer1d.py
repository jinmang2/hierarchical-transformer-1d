import sys
from inspect import isfunction

import torch
import torch.utils.checkpoint
from torch import nn

from einops import rearrange, repeat

from transformers.activations import ACT2FN
from transformerse import (
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


"""
@TODO
1. attention 코드 옮기기
2. reversible residual connection 적용하기
3. rotary 적용하기
"""


# lucidrains/rotary-embedding-torch
# ./rotary_embedding_torch/rotary_embedding_torch.py#L60
class HTransformer1DRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        theta = config.rotary_theta
        dim = config.hidden_size
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
        
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        
        if cache_key is not None:
            self.cache[cache_key] = freqs
            
        return freqs


class HTransformer1DEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
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
            
        return embeddings


class HTransformer1DSelfAttention(nn.Module):
    def __init__(self, config):
        embed_positions = HTransformer1DRotaryEmbedding(config)
        pass
    
    def forward(self):
        pass
    
    
# class ApplyNorm(nn.Module):
#     def __init__(self, fn, dim, is_post=False):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)
#         self.is_post = False
        
#     def forward(self, x, **kwargs):
#         if is_post:
#             x = self.fn(x, **kwargs)
#             x = self.norm(x)
#         else:
#             x = self.norm(x)
#             x = self.fn(x, **kwargs)
#         return x


# lucidrains/h-transformer-1d
# ./h_transformer_1d/h_transformer_1d.py#L73
class PreShiftToken(nn.Module):
    def __init__(self, fn, shifts):
        super().__init__()
        self.fn = fn
        self.shifts = shifts
    
    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)


class HTransformer1DAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = HTransformer1DSelfAttention(config)
        self.pruned_heads = set()
        
    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def forward(self):
        pass


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

    def forward(self, attention_output):
        return apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.output(hidden_states)
        return hidden_states        


class HTransformer1DLayer(nn.Module):
    def __init__(self, config, embed_positions=None):
        super().__init__()        
        self.attention = HTransformer1DAttention(config)  
        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None
        
        self.feed_forward  = ChunkHTransformer1DFeedForward(config)
        
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
    
#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         output_attentions=False,
#     ):        
#         self.attention_outputs = self.attention(
#             hidden_states,
#             attention_mask,
#             head_mask,
#             output_attentions=output_attentions,
#         )
#         attention_output = self_attention_outputs[0]
#         outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
#         layer_output = apply_chunking_to_forward(
#             self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
#         )
#         outputs = (layer_output,) + outputs
#         return outputs
    
#     def feed_forward_chunk(self, attention_output):
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output


class HTransformer1DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layer = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            layer.append(HTransformer1DLayer(config, embed_positions))
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            if self.gradient_checkpointing and self.training:
                
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
                
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
                
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
                
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
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
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, HTransformer1DSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HTransformer1DEncoder):
            module.gradient_checkpointing = value


class HTransformer1DModel(HTransformer1DPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = HTransformer1DEmbeddings(config)
            
        self.encoder = HTransformer1DEncoder(config)
        
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
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
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
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

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


class HTransformer1DForCausalLM(HTransformer1DPreTrainedModel):
    def __init__(self, config):
        pass
    
    def forward(self):
        pass


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