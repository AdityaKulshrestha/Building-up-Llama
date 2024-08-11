import torch 
from model import Llama
from transformers import AutoTokenizer
import habana_frameworks.torch.core as htcore
from torch.nn import functional as F 



model = Llama() 


model.load_state_dict(torch.load('ckpt_dir/model_10000_loss_5.593620300292969.pth'))

tokenizer = AutoTokenizer.from_pretrained('LingoIITGN/ganga-1b', trust_remote_code=True)
print(tokenizer.bos_token)
print(tokenizer.eos_token)
print(tokenizer.vocab_size)



# text = 'हिंदी एक सुंदर भाषा'
text = """
आपका नाम आदित्य है।
"""

tokenized_text = tokenizer(text, padding = 'max_length', max_length=512)['input_ids']


tokenized_text = torch.tensor([tokenized_text])
print(tokenized_text.shape)

# import sys 
# sys.exit()
tokenized_text = tokenized_text.to(torch.device('hpu'))
# print(tokenized_text)
max_new_tokens = 64
for i in range(max_new_tokens): 
    tokenized_text = tokenized_text[:, -512:]            # Seq Len/ Block Size 
    with torch.no_grad(): 
        output = model(tokenized_text) 
    last_output = output[:, -1,:]
    probs = F.softmax(last_output, dim=-1)
    idx_next = torch.argmax(probs, dim=None, keepdim=True)        #Not working good enough
    # idx_next = torch.multinomial(probs, num_samples=1)
    tokenized_text = torch.cat((tokenized_text, idx_next), dim=1)
    output_token = tokenizer.decode(idx_next.cpu().numpy()[0])
    print(output_token, end=" ") 





# Working for a single token
"""
with torch.no_grad():
    output =  model(tokenized_text)

print(output)
output = output[:, -1, :]           # Selecting the last generated token
print(output)

probs = F.softmax(output, dim = -1)
print(probs)
idx_next = torch.multinomial(probs, num_samples=1)
print(idx_next)
output_token= tokenizer.decode(idx_next.cpu().numpy()[0])
print(output_token)
            # logits = logits[:, -1, :] # become(B, C) 
            # apply softmax to get probabilities 
            # probs = F.softmax(logits, dim = -1)      # (B, C) 
            # sample from the distribution 
            # idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) 
            # append sampled index to the running sequence 
            # idx = torch.cat((idx, idx_next), dim = 1)  # (B, T+1)
            # print(decode(idx.cpu().numpy()[0]))
# output_text = tokenizer.decode(output)
"""

