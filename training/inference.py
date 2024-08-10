import torch 
from model import Llama
from transformers import AutoTokenizer
import habana_frameworks.torch.core as htcore
from torch.nn import functional as F 



model = Llama() 


model.load_state_dict(torch.load('ckpt_dir/model_510_loss_6.156224250793457.pth'))

tokenizer = AutoTokenizer.from_pretrained('LingoIITGN/ganga-1b', trust_remote_code=True)
print(tokenizer.bos_token)
print(tokenizer.eos_token)
print(tokenizer.vocab_size)



# text = 'हिंदी एक सुंदर भाषा'
text = """
यह महत्वपूर्ण है कि हम अपनी जीवनशैली में संतुलन बनाए रखें, क्योंकि एक संतुलित जीवन न केवल हमें शारीरिक रूप से स्वस्थ रखता है, बल्कि मानसिक और भावनात्मक स्थिरता को भी बनाए रखने में मदद करता है, जिससे हम जीवन की चुनौतियों का सामना अधिक आत्मविश्वास और सकारात्मक दृष्टिकोण के साथ कर सकते हैं।
आज के समय में जब तकनीकी प्रगति निरंतर हमारे जीवन के हर पहलू को बदल रही है, चाहे वह संचार के साधन हों, कार्यस्थल का स्वरूप हो, या फिर हमारे सामाजिक और व्यक्तिगत संबंध हों, यह अत्यंत आवश्यक हो गया है कि हम इन बदलावों के साथ तालमेल बैठाएं। इसके लिए न केवल व्यक्तियों को, 
बल्कि संगठनों और सरकारों को भी तेजी से अनुकूलन करना होगा, ताकि वे अपनी संबंधित क्षेत्रों में प्रतिस्पर्धी और प्रासंगिक बने रहें। इसके साथ ही, यह भी सुनिश्चित करना आवश्यक है कि इन प्रगति के नैतिक पहलुओं पर भी गहराई से विचार किया जाए। हमें इस बात का ध्यान रखना होगा कि नवाचार और प्रगति के बीच एक संतुलन बना रहे, 
ताकि हर व्यक्ति का भला हो सके और समाज में हर किसी के लिए समान अवसर उपलब्ध हों। यह भी जरूरी है कि तकनीक के लाभ समान रूप से सभी तक पहुंचें, चाहे वह व्यक्ति किसी भी सामाजिक-आर्थिक पृष्ठभूमि से आता हो, चाहे वह किसी भी भौगोलिक क्षेत्र में रहता हो, या उसकी शिक्षा का स्तर कुछ भी हो। अगर हम ऐसा करने में सफल होते हैं,
तो हम एक ऐसे भविष्य की कल्पना कर सकते हैं, जहां तकनीकी विकास न केवल समाज के कुछ खास वर्गों के लिए, बल्कि संपूर्ण मानवता के लिए लाभदायक सिद्ध हो।


यह महत्वपूर्ण है कि हम अपनी जीवनशैली में संतुलन बनाए रखें, क्योंकि एक संतुलित जीवन न केवल हमें शारीरिक रूप से स्वस्थ रखता है, बल्कि मानसिक और भावनात्मक स्थिरता को भी बनाए रखने में मदद करता है, जिससे हम जीवन की चुनौतियों का सामना अधिक आत्मविश्वास और सकारात्मक दृष्टिकोण के साथ कर सकते हैं।
आज के समय में जब तकनीकी प्रगति निरंतर हमारे जीवन के हर पहलू को बदल रही है, चाहे वह संचार के साधन हों, कार्यस्थल का स्वरूप हो, या फिर हमारे सामाजिक और व्यक्तिगत संबंध हों, यह अत्यंत आवश्यक हो गया है कि हम इन बदलावों के साथ तालमेल बैठाएं। इसके लिए न केवल व्यक्तियों को, 
बल्कि संगठनों और सरकारों को भी तेजी से अनुकूलन करना होगा, ताकि वे अपनी संबंधित क्षेत्रों में प्रतिस्पर्धी और प्रासंगिक बने रहें। इसके साथ ही, यह भी सुनिश्चित करना आवश्यक है कि इन प्रगति के नैतिक पहलुओं पर भी गहराई से विचार किया जाए। हमें इस बात का ध्यान रखना होगा कि नवाचार और प्रगति के बीच एक संतुलन बना रहे, 
ताकि हर व्यक्ति का भला हो सके और समाज में हर किसी के लिए समान अवसर उपलब्ध हों। यह भी जरूरी है कि तकनीक के लाभ समान रूप से सभी तक पहुंचें, चाहे वह व्यक्ति किसी भी सामाजिक-आर्थिक पृष्ठभूमि से आता हो, चाहे वह 
"""

tokenized_text = tokenizer(text)['input_ids']


tokenized_text = torch.tensor([tokenized_text])
print(tokenized_text.shape)

# import sys 
# sys.exit()
tokenized_text = tokenized_text.to(torch.device('hpu'))
# print(tokenized_text)
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




