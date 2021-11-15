from math import log2

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class HTransformer1DConfig(PretrainedConfig):
    r"""
    [DESCRIPTION]
    
    Args:
        
    Example::
    """
    model_type = "h-transformer-1d"
    
    def __init__(
        self,
        vocab_size=50000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        
        block_size=128, # this is the Nr in the paper - Nb = (max_seq_len / tokens_per_block)
        reversible=True, # use reversibility, to save on memory with increased depth
        shift_tokens=True, # whether to shift half the feature space by one along the sequence dimension, for faster convergence (experimental feature)
        
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=8192, # dim_head 정보로 parameterization
                                      # 얘는 max_seq_len으로 사용
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        attn_eps=1e-8,
        pad_token_id=0,
        rotary_value=False, # value도 rotary를 적용할 지 안할지
        rotary_theta=10000,
        learned_freq=False,
        # use_cache=True,
        position_embedding_type="rotary",
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        assert (max_position_embeddings % block_size) == 0, (
            'maximum sequence length must be divisible by the block size'
        )
        num_blocks = max_position_embeddings // block_size
        assert log2(max_position_embeddings // block_size).is_integer(), (
            f'number of blocks {num_blocks} must be a power of 2'
        )
        
        assert (hidden_size % num_attention_heads) == 0, (
            'hidden size must be divisible by the number of attention heads'
        )
        
        assert position_embedding_type in ['absolute', 'rotary'], (
            'position embedding type must be either \'absolute\' or \'rotary\''
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dim_head = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.reversible = reversible
        self.shift_tokens = shift_tokens
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.attn_eps = attn_eps
        self.rotary_value = rotary_value
        self.rotary_theta = rotary_theta
        self.learned_freq = learned_freq
        # self.use_cache = use_cache
        self.position_embedding_type = position_embedding_type