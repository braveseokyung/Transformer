from config import *
from preprocessing import *
from dataset import *
from layers import *

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=Transformer(src_vocab_size,tgt_vocab_size,d_model,max_seq_len,d_inner,num_heads).to(device)

def train(model,train_loader,optimizer,criterion):
    model.train()
    epoch_loss=0

    for batch_id, batch in enumerate(train_loader):
        src=batch.src
        tgt=batch.tgt
        out=model(src,tgt)
        
        optimizer.zero_grad()
        batch_loss=criterion(out,tgt)
        batch_loss.backward()
        optimizer.step()

        epoch_loss+=batch_loss.item()
        
    epoch_loss/=len(train_loader)    
    print(f"train loss: {epoch_loss}")
    return epoch_loss

def evaluate(model,test_loader,criterion):
    model.eval()
    epoch_loss=0

    with torch.no_grad():
        for batch_id,batch in enumerate(test_loader):
            src=batch.src
            tgt=batch.tgt
            out=model(src,tgt)

            batch_loss=criterion(out,tgt)
            epoch_loss+=batch_loss.item()

    epoch_loss/=len(test_loader)    
    print(f"eval loss: {epoch_loss}")
    return epoch_loss