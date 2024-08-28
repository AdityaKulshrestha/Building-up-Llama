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
from habana_frameworks.torch.hpex.optimizers import FusedAdamW

import logging

os.environ['LOG_LEVEL_ALL'] = '4'
os.environ['HABANA_LOGS']= '.habana_logs'

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Create logger
logger = logging.getLogger(__name__)

# Task - Change the pytorch optimizer 


config = {
    'train_iter': 1, 
    'save_dir': 'ckpt_dir', 
    'device': torch.device('hpu'), 
    'data_dir': 'data', 
    'batch_size': 16,
    'block_size': 128, 
    'min_lr': 3e-5,
    'max_lr': 3e-4,
    'save_freq': 100000, 
    'weight_decay': 1e-1, 
    'beta1': 0.9, 
    'beta2': 0.95, 
    'vocab_size': 64128
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
    # x, y = x.to(device), y.to(device)
    return x,y 


def train():
    model = Llama(vocab_size = config['vocab_size'], seq_len = config['block_size'])
    model = model.to(config['device'])
    print(sum(p.numel() for p in model.parameters())/1e9, 'Billion parameters')
    optimizer = FusedAdamW(model.parameters(), lr=config['min_lr'], betas=(config['beta1'], config['beta2']), eps=1e-08, weight_decay=config['weight_decay'])
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config['min_lr'], betas=(config['beta1'], config['beta2']), eps=1e-08, weight_decay=config['weight_decay'])
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config['min_lr']) 
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config['min_lr'], max_lr=config['max_lr'], step_size_up=10000, mode='exp_range')

    for iter in range(config['train_iter']):
        xb, yb = get_batch(config['data_dir'], 'train', config['batch_size'], config['block_size'], config['device'])

        for i, (x_i, y_i) in enumerate(tqdm(zip(xb, yb), total=xb.shape[0])):
            x_i, y_i = x_i.to(config['device']), y_i.to(config['device'])

            
            optimizer.zero_grad(set_to_none = True) 

            # with torch.autocast(device_type='hpu', dtype=torch.bfloat16):
            logits = model(x_i)
            B, L, C = logits.shape
            logits = logits.view(B*L, C)
            targets = y_i.view(B*L)
            # print(f"Logit shape : {logits.shape} Target Shape : {targets.shape}")
            loss = F.cross_entropy(logits, targets)
            current_lr = scheduler.get_last_lr()
            # print("Final Loss: ", loss.item())    

            print(f'Learning Rate: {current_lr[0]:2f}')
            print(f'Epoch {i + 1}, Final Loss: {loss.item():2f}')
            

            loss.backward() 
            htcore.mark_step()

            optimizer.step() 
            htcore.mark_step()

            scheduler.step()

            if i % config['save_freq'] == 0 and i!=0:
                x_val, y_val = get_batch(config['data_dir'], 'val', config['batch_size'], config['block_size'], config['device'])
                total_validation_loss = 0
                for _, (x_i_val, y_i_val) in enumerate(tqdm(zip(x_val, y_val), total=x_val.shape[0])):
                    x_i_val, y_i_val = x_i_val.to(config['device']), y_i_val.to(config['device'])
                    with torch.no_grad():
                        logits = model(x_i_val)
                        B, L, C = logits.shape
                        logits = logits.view(B*L, C)
                        targets = y_i_val.view(B*L)
                        val_loss = F.cross_entropy(logits, targets)
                        total_validation_loss += val_loss.item()




                torch.save(model.state_dict(), f'{config["save_dir"]}/model_{i}_loss_{total_validation_loss/x_val.shape[0]:2f}.pth')


    
            


if __name__ == "__main__":
    train()















