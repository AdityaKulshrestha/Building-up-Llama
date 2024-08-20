import os 
import torch 
import time 
import numpy as np
import torch.nn as nn 
from tqdm import tqdm 
from torch.nn import functional as F 
from transformers import AutoTokenizer
from datasets import load_dataset

import habana_frameworks.torch.hpu as ht
import habana_frameworks.torch.core as htcore

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
config = {
'vocab_size': 30000,
'dim': 4096, 
'num_layer': 8, 
'rms_eps': 1e-6,
'n_heads': 32,
'base': 10000, 
'scaling_factor': 1.0,
'seq_len': 512
}


def precompute_rotatory_embd(head_dim: int = 4096, seq_len: int = 512, device: str = torch.device('hpu'), theta: float = 10000.0):

    assert head_dim % 2 == 0, "Dimension must be divisible by two"

    # Formula theta_i = 10000 * (-2(i-1) / dim) for i = [1, 2, .... dim/2]

    theta_numerator = torch.arange(0, head_dim, 2).float()      # 1D Array (Head_dim / 2)

    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) 

    m = torch.arange(seq_len, device = device)       # Shape: (Seq_Len); 1D array
    # (Seq_Len) * (Head_Dim / 2) = (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin

    
def apply_customized_rope(q, k, cos, sin, position_ids):
    if q.device.type == "hpu" and has_fused_rope:
        # TODO: remove `.clone()` when it is fixed in SynapseAI
        if k.dtype == torch.bfloat16:
            return FusedRoPE.apply(
                q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
            ), FusedRoPE.apply(
                k,
                cos.unsqueeze(0).unsqueeze(0).clone().to(torch.bfloat16),
                sin.unsqueeze(0).unsqueeze(0).clone().to(torch.bfloat16),
                position_ids,
            )
        return FusedRoPE.apply(
            q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        ), FusedRoPE.apply(
            k, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        )
        

     


class Attention(nn.Module):

    def __init__(self,  dim: int, n_heads:int):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim 
        self.head_dim = self.dim // self.n_heads
        self.query = nn.Linear(dim, dim, bias = False)                         # Change the dimension, Add GQA 
        self.key = nn.Linear(dim, dim, bias = False)
        self.value = nn.Linear(dim, dim, bias = False)

        self.head_dim = self.dim // self.n_heads



    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor): #, query: torch.Tensor,key: torch.Tensor, value: torch.Tensor):
        
        # print(f"Cos shape : {cos.shape} Sin Shape: {sin.shape} ")
        batch_size, seq_len, _ = x.size()

        
        xq = self.query(x) 
        xk = self.query(x) 
        xv = self.value(x)
        # print(f"Shape after qkv multiplication Query: {xq.shape} Key: {xk.shape} Value {xv.shape}")


        xq, xk = apply_customized_rope(xq, xk, cos, sin, None)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # print("This is the attention block,", xq.shape)

        with ht.sdp_kernel(enable_recompute = True):
            sdpa_out = FusedSDPA.apply(xq, xk, xv, None, 0.1, True)
            
        sdpa_out = sdpa_out.transpose(1, 2).contiguous()
        # print("This is after transposing: ", sdpa_out.shape)
        sdpa_out = sdpa_out.view(batch_size, seq_len, self.dim)
        # print("This is the final sdpa output: ", sdpa_out.shape)
        return sdpa_out



class MLP(nn.Module): 

    def __init__(self, dim: int = 4096, hidden_dim: int = 9800):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False) 
        self.w2 = nn.Linear(hidden_dim, dim, bias = False) 
        self.w3 = nn.Linear(dim, hidden_dim, bias = False) 

    def forward(self, x: torch.Tensor): 
        swish = F.silu(self.w1(x)) 
        x_V = self.w3(x) 
        x = swish * x_V 
        x = self.w2(x) 
        # print(f"Final Shape of the tensor: {x.shape}")
        return x

class RMS(nn.Module): 
    """RMS Norm optimized for HPU"""
    def __init__(self, dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    
    def forward(self, x: torch.Tensor):
        hidden_states = FusedRMSNorm.apply(x, self.weight, self.eps)
        return hidden_states


class Decoder(nn.Module):

    def __init__(self, dim: int = 4096, vocab_size: int = 32000, n_heads: int = 16):
        super().__init__()
        self.attention = Attention(dim=dim, n_heads=n_heads)                                    # Change this head_size to the multihead size
        self.mlp = MLP()
        self.attn_rms = RMS()
        self.ffn_rms = RMS()


    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = self.attn_rms(x)
        # print("Shape after the RMS Norm : ", x.shape)
        x = x + self.attention.forward(x, cos, sin) 
        x = self.ffn_rms.forward(x)
        x = x + self.mlp.forward(x)
        return x 


        





class Llama(nn.Module):

    def __init__(self, vocab_size: int = 30000, dim: int = 4096, seq_len: int = 512, device: str = torch.device('hpu'), theta: float = 10000.0, n_layers: int = 8):
        super().__init__()
        self.vocab_size = vocab_size 
        self.dim = dim
        self.embeddings = nn.Embedding(self.vocab_size, self.dim)
        # self.decoder = Decoder()
        self.layers = nn.ModuleList(
            [Decoder() for _ in range(n_layers)]
        )
        self.cos , self.sin = precompute_rotatory_embd(dim, seq_len, device, theta)               # Introduced in transformer module so that we don't recompute everytime.
        self.lm_head = nn.Linear(dim, vocab_size)


    def forward(self, x: torch.Tensor):
        x = self.embeddings(x)
        # print("This is after embedding: ", x.shape)
        for layer in self.layers:
            x = layer.forward(x, self.cos, self.sin)
        # print("This is after decoder: ", x.shape)
        x = self.lm_head(x)
        return x



if __name__ == "__main__":

    # x = torch.tensor((4, 8), dtype=torch.int64)
    x = torch.randint(low=1, high=30000+1, size=(4, 8))
    x = x.to(torch.device('hpu'))
    model = Llama()
    output = model(x)
    print(output.shape)


    # rotatory = RotatoryPosEmbedding()
    # cos, sin = rotatory.precompute_rotatory_embd()
    # print(cos.shape, cos.dtype)
    # print(sin.shape, sin.dtype)

    # model = Llama()










    