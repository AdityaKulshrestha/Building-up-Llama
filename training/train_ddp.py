import os 
import torch 
from model import Llama
import logging
from tqdm import tqdm 
from torch.nn import functional as F 
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP 

from train import get_batch, get_batch_size
import torch.distributed as dist

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug
from habana_frameworks.torch.hpex.optimizers import FusedAdamW


os.environ['LOG_LEVEL_ALL'] = '4'
os.environ['HABANA_LOGS']= '~/.habana_logs'


logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Create logger
logger = logging.getLogger(__name__)


# Config 
config = {
    'train_iter': 1, 
    'eval_iter': 10, 
    'ckpt_iter': 20, 
    'save_dir': 'ckpt_dir', 
    'device': torch.device('hpu'), 
    'data_dir': 'data', 
    'batch_size': 16,
    'block_size': 512, 
    'lr': 3e-4,
    'save_freq': 10
}


# Implement DDP 
# Implement Autocast 
# Store these get_batch and get_batch_size in utils
# Implement optimized AdamW (giving error in graph build)

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


def train_ddp(rank, world_size):
    device = torch.device('hpu')
    setup(rank, world_size)
    # _, rank = init_distributed_mode()
    model = Llama().to(device)

    # optimizer = FusedAdamW(model.parameters(), lr = config['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr']) 


    parameters = count_parameters(model)
    print(f"Total Parameters: {parameters} B")

    if rank > -1:
        model = torch.nn.parallel.DistributedDataParallel(model)

    for iter in range(config['train_iter']): 
        xb, yb = get_batch(config['data_dir'], 'train', config['batch_size'], config['block_size'], config['device'])

        for i, (x_i, y_i) in enumerate(tqdm(zip(xb, yb), total=xb.shape[0])):
            optimizer.zero_grad(set_to_none=True) 

            with torch.autocast(device_type='hpu', dtype=torch.bfloat16):
                logits = model(x_i) 

                B, L, C = logits.shape
                logits = logits.view(B*L, C) 
                targets = y_i.view(B*L) 

                loss = F.cross_entropy(logits, targets) 

            loss.backward()
            htcore.mark_step()

            optimizer.step()
            htcore.mark_step()


            if i % config['save_freq'] == 0: 
                print(f"Iter : {i} Loss: {loss.item()}")
                torch.save(model.state_dict(), f'model_iter{i}_loss_{loss.item():2f}')


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