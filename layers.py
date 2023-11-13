import torch
import torch.nn as nn
import math

class FeedForward(nn.Module):
    def __init__(self,d_model=512,d_inner=2048):
        super().__init__()
        self.ffn=nn.Sequential(
            nn.Linear(d_model,d_inner),
            nn.ReLU(),
            nn.Linear(d_inner,d_model)
        )

    def forward(self,x):
        out=self.ffn(x)

        return out
    
# nn.Module 상속 : gradient를 흘려줘야 하니가 -> 여기서는 gradient 없지 않나?
# 나중에 Positional Encoding과 합치기
class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model=512):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.scale=math.sqrt(d_model)
    
    def forward(self,x):
        out=self.embedding(x)*self.scale

        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_len):
        super().__init__()

        self.PE=torch.zeros(shape=(d_model,max_seq_len))
        pos=torch.arrange(0,max_seq_len)
        _2i=torch.arrange(0,max_seq_len,2)

        self.PE[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.PE[:,1::2]=torch.sin(pos/(10000**(_2i/d_model)))

    def forward(self,input_embeds):

        batch_size, seq_len = input_embeds.size()
        out=input_embeds+self.PE[:seq_len,:]

        return out


