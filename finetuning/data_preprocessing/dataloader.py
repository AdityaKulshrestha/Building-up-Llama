from torch.utils.data import Dataset, DataLoader 
import torch 
from transformers import AutoTokenizer
import numpy as np 
from datasets import load_dataset



class CustomDataset(Dataset): 
    def __init__(self, dataset_name: str = "iamshnoo/alpaca-cleaned-hindi", tokenizer_name: str = "sarvamai/sarvam-2b-v0.5", ): 
        """
        Args: 
            dataset_name: The instruction pair dataset for the dataset preparation
            tokenizer_name: Name of the tokenizer to be used
        """

        dataset = load_dataset(dataset_name) 
        self.dataset = dataset['train']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = 
        self.user_instruction = """आप एक एआई सहायक हैं। आपका कार्य उपयोगकर्ताओं को सटीक और विस्तृत जानकारी प्रदान करके उनकी सहायता करना है। \n उपयोगकर्ता: {} \n एआई: \n"""

    def tokenize_inputs(self, text, max_length=64, strategy='left'):
        self.tokenizer.encode()



        
    def __len__(self): 
        return len(self.dataset)


    def __getitem__(self, idx):
        if self.dataset[idx]['input'] == "":            
            intruction = self.user_instruction.format(self.dataset[idx]['input'])
        response = self.dataset[idx]['output'] + ' ' + self.tokenizer.eos_token
        return None



if __name__ == "__main__": 
    data = CustomDataset()


