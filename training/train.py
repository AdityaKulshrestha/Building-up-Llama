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
from utils import LoadTextCorpus
from torch.utils.data import Dataset, DataLoader 


import logging

os.environ['LOG_LEVEL_ALL'] = '4'
os.environ['HABANA_LOGS']= '.habana_logs'

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Create logger
logger = logging.getLogger(__name__)


config = {
    'train_data': 'data/train.bin', 
    'val_data': 'data/val.bin',
    'train_iter': 1, 
    'save_dir': 'ckpt_dir', 
    'device': torch.device('hpu'), 
    'data_dir': 'data', 
    'batch_size': 8,
    'block_size': 2048, 
    'min_lr': 3e-5,
    'max_lr': 3e-4,
    'save_freq': 10000, 
    'weight_decay': 1e-1, 
    'beta1': 0.9, 
    'beta2': 0.95, 
    'vocab_size': 64128,
    'warmup_iters': 3000, 
    'lr_decay_iters': 600000 
}


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['max_lr'] * it / config['warmup_iters']
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return  config['min_lr'] + coeff * (config['max_lr'] - config['min_lr'])


def train():
    model = Llama(vocab_size = config['vocab_size'], seq_len = config['block_size'])
    model = model.to(config['device'])
    print(sum(p.numel() for p in model.parameters())/1e9, 'Billion parameters')

    optimizer = FusedAdamW(model.parameters(), lr=config['min_lr'], betas=(config['beta1'], config['beta2']), eps=1e-08, weight_decay=config['weight_decay'])
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config['min_lr'], betas=(config['beta1'], config['beta2']), eps=1e-08, weight_decay=config['weight_decay'])
        
    # Enabling cyclicLR for Leanring Rate
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config['min_lr'], max_lr=config['max_lr'], step_size_up=10000, mode='exp_range')

    # Load dataset 
    train_dataset = LoadTextCorpus(config['train_data'], config['block_size'])
    val_dataset = LoadTextCorpus(config['val_data'], config['block_size'])

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True, num_workers=8, drop_last=True)    
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle=True, num_workers=8, drop_last=True)      

    for it in range(config['train_iter']):
        t = tqdm(range(len(train_dataloader)), desc="Training", unit="iteration")
        for i, (x_i, y_i) in enumerate(train_dataloader):
            x_i, y_i = x_i.to(config['device']), y_i.to(config['device'])


            lr = get_lr(i)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            

            # with torch.autocast(device_type='hpu', dtype=torch.bfloat16):
            logits = model(x_i)
            B, L, C = logits.shape
            logits = logits.view(B*L, C)
            targets = y_i.view(B*L)

            loss = F.cross_entropy(logits, targets)
            # current_lr = scheduler.get_last_lr()



            # print(f'Learning Rate: {current_lr[0]:2f}')
            # print(f'Epoch {i + 1}, Final Loss: {loss.item():2f}')
            t.set_postfix({'Epoch': f"{i + 1}", 'Final Loss': f"{loss.item():2f}", 'Learning Rate': f'{lr:2f}'})
            

            loss.backward() 
            htcore.mark_step()

            optimizer.step() 
            htcore.mark_step()

            optimizer.zero_grad(set_to_none = True) 

            # scheduler.step()

            if i % config['save_freq'] == 0 and i!=0:
                total_validation_loss = 0
                for _, (x_i_val, y_i_val) in enumerate(tqdm(val_dataloader, total=len(val_dataloader))):
                    x_i_val, y_i_val = x_i_val.to(config['device']), y_i_val.to(config['device'])
                    with torch.no_grad():
                        logits = model(x_i_val)
                        B, L, C = logits.shape
                        logits = logits.view(B*L, C)
                        targets = y_i_val.view(B*L)
                        val_loss = F.cross_entropy(logits, targets)
                        total_validation_loss += val_loss.item()




                torch.save(model.state_dict(), f'{config["save_dir"]}/model_{i}_loss_{total_validation_loss/len(val_dataloader):2f}.pth')


    
            


if __name__ == "__main__":
    train()















