import os 
from tqdm import tqdm 
import numpy as np 
from transformers import AutoTokenizer
from datasets import load_dataset 



# number of workers in .map() call 
# good number to use is ~order number of cpu cores // 2
num_proc = 8 

# number of workers in load_dataset() call 
# best number might be different from num_proc above as it also depends on NW speed. 
# it is better than 1 usually though 

num_proc_load_dataset = num_proc 

model_id = 'sarvamai/OpenHathi-7B-Hi-v0.1-Base'
dataset_name = 'shivam/hindi_pib_processed'
tokenizer = AutoTokenizer.from_pretrained(model_id)
bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token


def process(example): 
    text = example['text']+eos_token        # bos itself gets appended in the tokenizer!
    ids = tokenizer(text)['input_ids']
    out = {'ids': ids, 'len': len(ids)}
    return out 



if __name__ == '__main__':
    dataset = load_dataset(dataset_name, num_proc = num_proc_load_dataset)

    # hindi corpus by default containes only train split, so create a test split 
    split_dataset = dataset['train'].train_test_split(test_size = 0.05, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')        # rename the test split to val 


    tokenized_dataset = split_dataset.map(
        process, 
        remove_columns=['text'], 
        desc='tokenizing the split', 
        num_proc = num_proc, 
    )

    for split, dset in tokenized_dataset.items(): 
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16       # saving datatype and hence data 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len), )
        total_batches = 1024 

        idx = 0 
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write 
            batch = dset.shard(num_shards=total_batches, index = batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])

            # write into mmap 
            arr[idx: idx+len(arr_batch)] = arr_batch 
            idx+= len(arr_batch) 
        arr.flush() 

