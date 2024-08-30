import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 


class LoadTextCorpus(Dataset): 
    def __init__(self, data_path, seq_length): 
        """
        Args: 
            data_path (string): Path to the dataset file 
            seq_length (int): Length of the sequence or context length 
        """
        self.seq_length = seq_length 

        # Open the memory-mapped file 
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

        # Calculate the number of sequences that can be created 
        self.num_sequences = len(self.data) // self.seq_length 

    def __len__(self): 
        return self.num_sequences 

    def __getitem__(self, idx): 
        # Get the sequences starting at the idx 
        start = idx * self.seq_length 
        end = start + self.seq_length 
        x_seq = self.data[start: end]
        y_seq = self.data[start+1: end+1]


        # Convert to tensor 
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(y_seq, dtype=torch.long)

if __name__ == "__main__": 

    data_path = "data/train.bin"
    seq_length = 128
    batch_size = 16 

    dataset = LoadTextCorpus(data_path, seq_length)

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=8, drop_last=True)      # Drops the last batch if the size is smaller than the given
    
    # Printing total dataset length
    print(len(dataloader))

    # Iterating over each dataset 
    # for i in dataloader: 
        # print(i.size())

    # Printing a sample data instance
    x, y = next(iter(dataloader))
    print(x.shape)
    print(x)
    print(y.shape)
    print(y)