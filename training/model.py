import os 
import torch 
import time 
import numpy as np
import torch.nn as nn 
from tqdm import tqdm 
from torch.nn import functional as F 
from transformers import AutoTokenizer
from datasets import load_dataset

try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE

    has_fused_rope = True
except ImportError:
    has_fused_rope = False
    print("Not using HPU fused kernel for apply_rotary_pos_emb")


try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm

    has_fused_rms_norm = True
except ImportError:
    has_fused_rms_norm = False
    print("Not using HPU fused kernel for RMSNorm")

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

import habana_frameworks.torch.core as htcore


# Model Architecture 
"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
"""

# Model parameters 
vocab_size = 30000 
dim = 4096 
num_layer = 16 
rms_eps = 1e-6



class Attention(nn.Module):

    def __init__(self, ):
        super().__init__()


    def forward():



class MLP(nn.Module): 

    def __init__(self, ):
        super().__init__()


class RMS(nn.Module): 
    """RMS Norm optimized for HPU"""
    def __init__(self, dim:int , eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    
    def forward(self, x: torch.Tensor):
        hidden_states = FusedRMSNorm.apply(hidden_states, self.weight, self.eps)
        return hidden_states


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention = Attention()
        self.mlp = MLP()
        self.attn_rms = RMS()
        self.ffn_rms = RMS()


    def forward(self):



class LLama(nn.Module):

    def __init__(self, vocab_size: int = 30000, dim: int = 4096):
        super().__init__()

        self.embeddings = nn.Embedding(self.vocab_size, self.dim)






if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    print(model)

    