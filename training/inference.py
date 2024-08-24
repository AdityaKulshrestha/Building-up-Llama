import torch 
from model import Llama
from transformers import AutoTokenizer
import habana_frameworks.torch.core as htcore
from torch.nn import functional as F 


config = {
    'vocab_size': 64128, 
    'block_size': 128
}

model = Llama(vocab_size=config['vocab_size'], seq_len = config['block_size']) 


model.load_state_dict(torch.load('ckpt_dir/model_5000_loss_6.356828.pth'))

tokenizer = AutoTokenizer.from_pretrained('sarvamai/sarvam-2b-v0.5', trust_remote_code=True)
print(tokenizer.encode(tokenizer.bos_token))
print(tokenizer.eos_token)
print(tokenizer.vocab_size)



# text = 'हिंदी एक सुंदर भाषा'
# text = """आपका नाम आदित्य"""
text = '''पश्चिम बंगाल के राज्यपाल स्व० एच० मी० मुखर्जी एवं लोकसभा के अध्यक्ष स्व० अनन्त शायनम आयगर के साथ श्री सीतारामजी उन्होंने विश्वविद्यालय में शिक्षा नही पाई, फिर भी बगाल की तेजस्विता के बीच अपने को प्रतिष्ठित किया है और मुक्ति आन्दोलन के महान् कर्णधारो का अजस्र स्नेह पाया है। जिस समय राजनीति का अर्थ सेवा था, उस समय वे सब से आगे थे । आज राजनीति भोग-नीति है, इसलिए उनमे न पद के लिए होड है, न सम्मान के प्रति आसक्ति अपनी जीवन सध्या में भी वे जन-कल्याण के लिए, साहित्य और कला के लिए समर्पित है। इस आयु मे भी श्री शिक्षायतन जैसी संस्थाए उनकी क्रियात्मक शक्तियो के केन्द्र उन्होंने अपनी गलतियो का बोझ कभी नही ढोया है । उनका प्रयोग ऊपर चढने के लिए सीढी के रूप में ही किया है । गाधीवाद मे जो सर्वोत्तम है, उसका वे प्रतीक है । वे 'माँ' का रूप हैं, 'माँ', जो बस प्यार ही कर सकती है -- जिसके लिए सार्थक होने का अर्थ ही समर्पित होना है । समर्पण की इसी चमत्कारी प्राभा से मण्डित हे सौम्य-मूर्ति सेकसरियाजी का सम्पूर्ण जीवन । बगला के ख्याति प्राप्त नाट्यकार, कहानीकार और उपन्यास लेखक श्री तरुण राय मरूद्यान का यह मानव ! मुझे मरुभूमि का कोई अनुभव नहीं है क्योकि वहाँ जाने का कभी मौका ही नही मिला । हाँ, मरुभूमि की तस्वीर देखी है, वर्णन पढा है । पच्चीस साल पहले जब कालेज से निकला ही था, रोजगार के लिये दलाली का जुआ कधे पर रख डलहौजी स्क्वायर मे जाने लगा था। यह क्षेत्र एक बडा फैला हुआ रेगिस्तान ही है - रस रूप हीन सूखा खखाड । मै साहित्य का विद्यार्थी था, नाटक करना मुझे अच्छा लगता था, लेकिन इन सब की बातें करने के लिये इन आफिसो के वातावरण मे कही कोई नहीं था । वहा तो फाटका बाजार की चिल्लाहट, शेयर के ऊँचेनीचे भाव, पाट के दाम, चट की दरें, जहाज का किराया, -- इन्ही सब की कच कच थी । सुना है कि ऊँट मरुस्थल मे काटेदार वृक्षो की डालियाँ चवाना पसन्द करता है । मुह से खून निकलता है, फिर भी उसी मे उसे । मै भी यदि डलहौजी स्क्वायर की मरुभूमि का ऊँट वन पाता तो, हो सकता है, औरो की तरह रुपये के कँटीले पौधे चबा कर आनन्द पाता किन्तु दुर्भा यवश वैसा मैं नहीं बन पाया । इमलिये इस मरुभूमि में एक स्निग्ध-शीतल मरुद्यान की खोज करता रहता था । आखिर मरुद्यान मिल गया । मेरे परम मित्र श्री बी० एम० सिंधी के माध्यम से एक दिन मुझे मनचाहा मरुद्यान मिल गया । वह था 'नया समाज' नामक प्रगतिवादी हिन्दी मासिक पत्रिका का दफ्तर जिस दफ्तर मे सिंघीजी काम करते थे, उसी के एक कोने मे उस पत्रिका का दफ्तर था और उन्ही की देखरेख में उसका सचालन था । सम्पादक थे श्री मोहनसिंह सेंगर, जो दुर्भाग्य से आज जीवित नही है । वे समर्थ लेखक तो थे ही, किन्तु उससे भी बडे वे मानव थे । भारतीय संस्कृति का पूर्ण विकास मैने उनमे पाया था । इसीलिये मरुभूमि की बालू की गर्मी मे तप्त होते ही भाग कर मै प्राश्रय लेता था "नया समाज" के इस मरुद्यान मे । मन खोल कर सेगरजी से बाते करता - साहित्य, कला, नाटक आदि के बारे मे । हमारी इन बैठको मे और एक व्यक्ति प्राय उपस्थित रहता था - सौम्य - कान्ति, लम्बा, गोरा, खादी का सफेद कुर्ता-धोती पहने और माथे पर गांधी टोपी लगाये । मुह पर अशेप हँसी । विनयी, नम्र, धीर, स्थिर । मुस्कराते हुए हमारी बातें सुनता,'''

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenized_text = tokenizer(text, padding = 'max_length', max_length=128,)['input_ids']


tokenized_text = torch.tensor([tokenized_text])
print(tokenized_text.shape)

# import sys 
# sys.exit()
tokenized_text = tokenized_text.to(torch.device('hpu'))
# print(tokenized_text)
max_new_tokens = 64
for i in range(max_new_tokens): 
    tokenized_text = tokenized_text[:, -128:]            # Seq Len/ Block Size 
    # print(tokenized_text)
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

