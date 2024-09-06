import os 
import torch 
from model import Llama
import logging
from tqdm import tqdm 
from torch.nn import functional as F 
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import Dataset, DataLoader 

import torch.distributed as dist

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug
from habana_frameworks.torch.hpex.optimizers import FusedAdamW

from utils import LoadTextCorpus


os.environ['LOG_LEVEL_ALL'] = '4'
os.environ['HABANA_LOGS']= '.habana_logs'


logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Create logger
logger = logging.getLogger(__name__)


# Config 
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
    'weight_decay': 1e-1, 
    'beta1': 0.9, 
    'beta2': 0.95, 
    'save_freq': 10000, 
    'intermediate_dim': 11008,
    'vocab_size': 64128, 
}


# Implement Autocast; Not working giving specific operation issue in SDPA GEMM; Not supported for bf16
# Implement optimized AdamW (giving error in graph build)

# Deprecated
def init_distributed_mode():
    world_size = 0 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    import habana_frameworks.torch.distributed.hccl 
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu 

    world_size, rank, local_rank = initialize_distributed_hpu()
    print(f"World Size {world_size} Rank {rank} Local Rank {local_rank}")
    
    process_per_node = 8 
    if world_size > 1: 
        os.environ['MAX_WAIT_ATTEMPTS'] = '50' 
        torch.distributed.init_process_group('hccl', rank=rank, world_size=world_size)

    return world_size, rank

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    #distributed package for HCCL
    import habana_frameworks.torch.distributed.hccl
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

    # torch.distributed.

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000000

def save_checkpoint(model, optimizer, iteration, loss, filename):
    checkpoint = {
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss
    }
    torch.save(checkpoint, filename)


def train_ddp(rank, world_size):
    setup(rank, world_size)
    # _, rank = init_distributed_mode()
    model = Llama(vocab_size = config['vocab_size'], seq_len = config['block_size'])
    model = model.to(config['device'])

    # optimizer = FusedAdamW(model.parameters(), lr = config['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['min_lr'], betas=(config['beta1'], config['beta2']), eps=1e-08, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config['min_lr'], max_lr=config['max_lr'], step_size_up=10000, mode='exp_range')

    parameters = count_parameters(model)
    if rank==0:         # Prints it only one time
        print(f"Total Parameters: {parameters} B")

    train_dataset = LoadTextCorpus(config['train_data'], config['block_size'])
    val_dataset = LoadTextCorpus(config['val_data'], config['block_size'])

    if rank > -1:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank)

        train_loader = DataLoader(train_dataset, sampler = train_sampler, batch_size = config['batch_size'], num_workers=8, drop_last=True)
        val_loader = DataLoader(val_dataset, sampler = val_sampler, batch_size = config['batch_size'], num_workers=8, drop_last = True)
        

    for iter in range(config['train_iter']): 

        for i, (x_i, y_i) in enumerate(train_loader):
            x_i, y_i = x_i.to(config['device']), y_i.to(config['device'])

            optimizer.zero_grad(set_to_none=True) 

            # with torch.autocast(device_type='hpu', dtype=torch.bfloat16):
            logits = model(x_i) 

            B, L, C = logits.shape
            logits = logits.view(B*L, C) 
            targets = y_i.view(B*L) 

            loss = F.cross_entropy(logits, targets) 

            loss.backward()
            htcore.mark_step()

            optimizer.step()
            htcore.mark_step()

            if rank==0:         # Prints it only one time
                print(f"Iter : {i} Loss: {loss.item()}")

            if (i % config['save_freq'] == 0) and i != 0 and rank == 0: 
                save_checkpoint(model, optimizer, i, loss.item(), 
                                f'{config["save_dir"]}/model_DDP_iter{i}_loss_{loss.item():.2f}.pt')

        cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__": 

    # How to run 
    # mpirun --allow-run-as-root -np 2 python train_ddp.py
    world_size = 2
    run_demo(train_ddp, world_size)    
    # train_ddp()

