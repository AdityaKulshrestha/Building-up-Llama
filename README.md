# Building up Llama




## Architectural Modifications in Llama 

### RoPE
RoPE stands for Rotatory Positional Embedding. 

Previous solution for positional encoding:
- Learning from Data 
- Sinusoidal functions

Why RoPE was introduced?
- Original positional encoding methods were limited in sequence length due to the fact that they were trained on limited size data. 
- The absolute positional encoding (sinusoidal encodings) didn't contain the information about the relative position of tokens. 

#### Original Attention Mechanism
Absolute positional encoding 
$$
\text{PE}_{\text{pos}, 2i} = \sin \left( \frac{\text{pos}}{10000^{2i/d_{\text{model}}}} \right)$$

$$\text{PE}_{\text{pos}, 2i+1} = \cos \left( \frac{\text{pos}}{10000^{2i/d_{\text{model}}}} \right)$$
_where i is the index of the token and is the dimension of the embedding._

**Final Embeddings = Original Embeddings + Absolute Positional Encoding** 

#### Attention Mechanism in RoPE
Query Vector = $(W_{q}x_{m})e^{im\theta}$

Key Vector = $(W_{k}x_{n})e^{in\theta}$



## Checklist 

- [x] Inference script for Llama 
- [x] Model initialization
- [x] Embedding Module 
- [x] RMS Module 
- [ ] Attention Module 
