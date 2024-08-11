import os 
import torch 
import numpy as np
import torch.nn as nn 
from tqdm import tqdm 
from torch.nn import functional as F 
from transformers import AutoTokenizer
import habana_frameworks.torch.core as htcore
from datasets import load_dataset
from model import Llama 

import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Create logger
logger = logging.getLogger(__name__)

# Task - Change the pytorch optimizer 


config = {
    'train_iter': 1, 
    'eval_iter': 10, 
    'ckpt_iter': 20, 
    'save_dir': 'ckpt_dir', 
    'device': torch.device('hpu'), 
    'data_dir': 'data', 
    'batch_size': 32,
    'block_size': 512, 
    'lr': 3e-4,
    'save_freq': 10000
}


def get_batch_size(length, batch_length, cntx_len):
    total_batches = int(length / (batch_length*cntx_len))
    return total_batches


def get_batch(data_dir, split, batch_size, block_size, device): 
    # We recreate np.memmap every batch to avoid a memory leak
    logger.info("Processing Dataset!")
    if split=='train': 
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        total_batch = get_batch_size(len(data), batch_size, block_size)           # Default 440

    else: 
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        total_batch = get_batch_size(len(data), batch_size, block_size)           # Default 440 


    ix = torch.randint(len(data) - block_size, (len(data)//block_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in tqdm(ix)])
    x = x[:total_batch*batch_size, :]       # HARDCODED RIGHT NOW!
    x = x.view(-1, batch_size, block_size)
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in tqdm(ix)])
    y = y[:total_batch*batch_size, :]       # HARDCODED RIGHT NOW!
    y = y.view(-1, batch_size, block_size)
    x, y = x.to(device), y.to(device)
    return x,y 


def train():
    model = Llama()
    model = model.to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr']) 

    for iter in range(config['train_iter']):
        xb, yb = get_batch(config['data_dir'], 'train', config['batch_size'], config['block_size'], config['device'])

        for i, (x_i, y_i) in enumerate(tqdm(zip(xb, yb), total=xb.shape[0])):
            
            optimizer.zero_grad(set_to_none = True) 

            logits = model(x_i)
            # print("Output Shape", logits.shape)
            # print("Target Size: ", y_i.shape) 
            B, L, C = logits.shape
            logits = logits.view(B*L, C)
            targets = y_i.view(B*L)
            print(f"Logit shape : {logits.shape} Target Shape : {targets.shape}")
            loss = F.cross_entropy(logits, targets)
            print("Final Loss: ", loss.item())
            

            loss.backward() 
            htcore.mark_step()

            optimizer.step() 
            htcore.mark_step()

            if i % config['save_freq'] == 0:
                torch.save(model.state_dict(), f'{config["save_dir"]}/model_{i}_loss_{loss.item():2f}.pth')


    
            


if __name__ == "__main__":
    train()















