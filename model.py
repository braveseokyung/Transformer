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
    
# nn.Module 상속 : gradient를 흘려줘야 하니가
# 이걸 class로 만들 필요가 있을까?
class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model=512):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.scale=math.sqrt(d_model)
    
    def forward(self,x):
        out=self.embedding(x)*self.scale

        return out
    
class PositionalEncoding(nn.Module):