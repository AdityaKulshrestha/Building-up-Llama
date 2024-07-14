import torch 
from llama_2 import GPTLanguageModel
from transformers import AutoTokenizer
import habana_frameworks.torch.core as htcore


model = GPTLanguageModel() 
model.load_state_dict(torch.load('hindi_llama.pth'))
tokenizer = AutoTokenizer.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base')



text = "नमस्ते आप "

tokenized_text = tokenizer(text)['input_ids']
tokenized_text = torch.tensor([tokenized_text])
tokenized_text = tokenized_text.to(torch.device('hpu'))
print("Tokenized Text: ",tokenized_text)


output =  model.generate(tokenized_text, max_new_tokens=5)[0].tolist()
print("Generated tokens: ", output)

output_text = tokenizer.decode(output)
print("Decoded output text: ", output_text)






