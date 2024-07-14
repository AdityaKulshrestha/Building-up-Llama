import os 
import torch 
import numpy as np
import torch.nn as nn 
from tqdm import tqdm 
from torch.nn import functional as F 
from transformers import AutoTokenizer
import habana_frameworks.torch.core as htcore
from datasets import load_dataset



# hyperparameters 
batch_size = 128 
block_size = 512        # Max context length OR sequence length 
max_iters = 50 
eval_interval = 5 
learning_rate = 3e-4 
device = torch.device('hpu')
eval_iters = 500
n_embd = 2048 
n_head = 6 
n_layer = 6 
dropout = 0.2 


# data configs 
data_dir = 'prepare_dataset'

torch.manual_seed(1337)



# Loaded the tokenizer 

tokenizer = AutoTokenizer.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base')
bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size



# data loading 
def get_batch(split): 
    # We recreate np.memmap every batch to avoid a memory leak

    if split=='train': 
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    else: 
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x,y 


X, Y = get_batch('train')


@torch.no_grad()
def estimate_loss(): 
    out = {}
    model.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): 
            X, Y = get_batch(split) 
            logits, loss = model(X, Y)
            losses[k] = loss.item() 
        out[split] = losses.mean() 
    model.train()
    return out 



class Head(nn.Module):
    """ one head of self-attention"""
    def __init__(self, head_size): 
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False) 
        self.query = nn.Linear(n_embd, head_size, bias = False) 
        self.value = nn.Linear(n_embd, head_size, bias = False) 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x): 
        # input of size (batch, time-step, channels) 
        # output of size (batch, time-step, head size) 
        B, T, C = x.shape 
        k = self.key(x)     # (B, T, hs) 
        q = self.query(x)   # (B, T, hs) 
        # compute attention scores ("affinities") 
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # Scaling out the attention # (B, T, hs)    @ (B, hs, T)    -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T) 
        wei = F.softmax(wei, dim = -1)      # (B, T, T) 
        wei = self.dropout(wei) 
        # perform the weighted aggregation of the values 
        v = self.value(x)   # (B, T, hs) 
        out = wei @ v # (B, T, T) @ (B, T, hs)      -> (B, T, hs) 
        return out 
    


class MultiHeadAttention(nn.Module): 
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size): 
        super().__init__() 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) 
        self.dropout = nn.Dropout(dropout) 


    def forward(self, x): 
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out)) 
        return out 
    

class FeedForward(nn.Module): 
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd): 
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(), 
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        return self.net(x) 
    

class Block(nn.Module): 
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head): 
        # n_embd: embedding dimension, n_head: the number of heads we'd like 
        super().__init__() 
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size) 
        self.ffwd = FeedForward(n_embd)  
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd) 


    def forward(self, x): 
        x = x + self.sa(self.ln1(x))    # Layer norm is applied first then passed to the self attention
        x = x + self.ffwd(self.ln2(x))     # Layer norm is again applied first then passed to the MLP layer 
        return x 
    

class GPTLanguageModel(nn.Module): 

    def __init__(self): 
        super().__init__() 
        # each token directly reads off the logits for the next token from a lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)    # final layer norm 
        self.lm_head = nn.Linear(n_embd, vocab_size) 

        # better init not covered in this; check it out 
        self.apply(self._init_weights) 

    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02) 
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias) 
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02) 


    def forward(self, idx, targets = None): 
        B, T = idx.shape 

        # idx and targets are both (B, T) tensor of integer 
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))     # (T, C) 
        x = tok_emb + pos_emb # (B,T, C) 
        x = self.blocks(x)  # (B, T, C) 
        x = self.ln_f(x)    # (B, T, C) 
        logits = self.lm_head(x)   # (B, T, vocab_size) 


        if targets is None: 
            loss = None 

        else: 
            B, T, C = logits.shape 
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets) 

        return logits, loss 
    

    def generate(self, idx, max_new_tokens): 
        # idx is (B, T) array of indices in the current context 
        for _ in range(max_new_tokens): 
            # crop idx to the last block_size tokens 
            idx_cond = idx[:, -block_size:]
            # get the predictions 
            logits, loss = self(idx_cond) 
            # focus only on the last time step 
            logits = logits[:, -1, :] # become(B, C) 
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim = -1)      # (B, C) 
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) 
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim = 1)  # (B, T+1)
        return idx.cpu() 
    

model = GPTLanguageModel() 
m = model.to(device) 
# print the number of parameters in the model 
print(sum(p.numel() for p in m.parameters())/1e6, 'million parameters')


# create a pytorch optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 


for iter in range(max_iters): 

    # every once in a while evaluate the loss on train and val sets 
    if iter % eval_interval == 0 or iter == max_iters -1: 
        losses = estimate_loss() 
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    # sample a batch of data 
    xb, yb = get_batch('train')

    # evaluate the loss = 
    logits, loss = model(xb, yb) 
    optimizer.zero_grad(set_to_none = True) 

    loss.backward() 
    htcore.mark_step()

    optimizer.step() 
    htcore.mark_step()

device = torch.device('cpu')
torch.save(model.state_dict(), 'hindi_llama.pth', device=device)

