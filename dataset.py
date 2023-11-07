from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import pickle
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self,en,de):
        self.en=torch.tensor(en)
        self.de=torch.tensor(de)
    def __len__(self):
        return len(self.en)
    def __getitem__(self, index):
        return self.en[index], self.de[index]

def custom_collate_fn(batch):
    en, de = list(zip(*batch))
    padded_en = nn.utils.rnn.pad_sequence(en)
    padded_de = nn.utils.rnn.pad_sequence(de)
    
    return [padded_en,padded_de]

train_df=pd.read_pickle("./Dataset/train_df.pkl")
test_df=pd.read_pickle("./Dataset/val_df.pkl")
val_df=pd.read_pickle("./Dataset/test_df.pkl")

train_dataset=CustomDataset(train_df['en_encoded'],train_df['de_encoded'])
val_dataset=CustomDataset(val_df['en_encoded'],val_df['de_encoded'])
test_dataset=CustomDataset(test_df['en_encoded'],test_df['de_encoded'])

train_dataloader=DataLoader(train_dataset,batch_size=16,shuffle=True,collate_fn=custom_collate_fn)
val_dataloader=DataLoader(val_dataset,batch_size=16,shuffle=False,collate_fn=custom_collate_fn)
test_dataloader=DataLoader(test_dataset,batch_size=16,shuffle=False,collate_fn=custom_collate_fn)