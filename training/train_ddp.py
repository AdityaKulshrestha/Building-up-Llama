from model import Llama
import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Create logger
logger = logging.getLogger(__name__)



# Implement DDP 
# Implement Autocast 

def init_distributed_mode():
    world_size = 0 

    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu 
    world_size, rank, local_rank = initialize_distributed_hpu()
    print(f"World Size {world_size} Rank {rank} Local Rank {local_rank}")



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000000


def train_ddp():
    init_distributed_mode()
    model = Llama()
    parameters = count_parameters(model)
    print(f"Total Parameters: {parameters} B")


if __name__ == "__main__": 
    train_ddp()