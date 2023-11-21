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
        pos=torch.arange(0,max_seq_len)
        _2i=torch.arange(0,max_seq_len,2)

        self.PE[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.PE[:,1::2]=torch.sin(pos/(10000**(_2i/d_model)))

    def forward(self,input_embeds):

        batch_size, seq_len = input_embeds.size()
        out=input_embeds+self.PE[:seq_len,:]

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h):
        super().__init__()
        self.num_heads=h
        self.d_model=d_model
        self.d_k=d_model//h # d_key=d_value
        self.query_linear=nn.Linear(d_model,d_model)
        self.key_linear=nn.Linear(d_model,d_model)
        self.value_linear=nn.Linear(d_model,d_model)
        self.out_linear=nn.Linear(d_model,d_model)

    def split_to_multi_heads(self,x):
        batch_size,seq_len,d_model=x.size()
        splitted_x=x.reshape(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)

        return splitted_x

    def scaled_dot_product_attention(self,Q,K,V):
        att_score=torch.dot(Q,torch.transpose(K,-2,-1)) # b*
        scale_factor=math.sqrt(self.d_key)
        cos_similarity=att_score/scale_factor
        self_attention=torch.matmul(nn.Softmax(cos_similarity),V)
            
        return self_attention
    
    def concat_multi_heads(self,x):
        batch_size,num_heads,seq_len,d_k=x.size()
        concat_x=x.transpose(1,2).reshape(batch_size,seq_len,self.d_model)

        return concat_x


    def forward(self,Q,K,V,mask):
        # input embedding+positional encoding
        Q_heads=self.split_to_multi_heads(self.query_linear(Q))
        K_heads=self.split_to_multi_heads(self.key_linear(K))
        V_heads=self.split_to_multi_heads(self.value_linear(V))

        multi_attention=self.scaled_dot_product_attention(Q_heads,K_heads,V_heads)
        concat_attention=self.concat_multi_heads(multi_attention)
        out=self.out_linear(concat_attention)

        return out


class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_inner,num_heads):
        super().__init__()

        # self.Emb=Embedding(vocab_size,d_model)
        # self.PE=PositionalEncoding(d_model,max_seq_len)
        self.MHA=MultiHeadAttention(d_model,num_heads)
        self.FF=FeedForward(d_model,d_inner)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

    def forward(self,x):
        # 첫번째 encoder에서 embedding layer를 거친 값
        # emb=self.PE(self.Emb(x))
        attn_out=self.MHA(x,x,x)
        norm_out=self.norm1(x+attn_out)
        ff_out=self.FF(norm_out)
        out=self.norm2(ff_out+out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_inner,num_heads):
        super().__init__()
        self.dec_MHA=MultiHeadAttention(d_model,num_heads)
        self.enc_dec_MHA=MultiHeadAttention(d_model,num_heads)
        self.norm1=nn.Layernorm(d_model)
        self.norm2=nn.Layernorm(d_model)
        self.norm3=nn.Layernorm(d_model)

    def forward(self,x,encoder_out,src_mask,target_mask):
        dec_attn_out=self.dec_MHA(x,x,x,target_mask)
        norm_out=self.norm1(x+dec_attn_out)
        enc_dec_attn_out=self.enc_dec_MHA(encoder_out,encoder_out,norm_out,mask=None)
        norm_out_2=self.norm2(norm_out+enc_dec_attn_out)
        ff_out=self.ff(norm_out_2)
        out=self.norm3(norm_out_2+ff_out)

        return out
