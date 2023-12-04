import sentencepiece as spm
from datasets import load_dataset
import pandas as pd
import pickle

dataset = load_dataset("iwslt2017",'iwslt2017-en-de')

def dict_to_df(dict):
    df=pd.DataFrame()
    en=[]
    de=[]
    for it in dict:
        en.append(it['translation']['en']+"\n")
        de.append(it['translation']['de']+"\n")

    df['en']=pd.DataFrame(en)
    df['de']=pd.DataFrame(de)

    return df

def tokenize(df,en_sp,de_sp):
    df['en_encoded']=df['en'].apply(lambda x:en_sp.encode(x))
    df['de_encoded']=df['de'].apply(lambda x:de_sp.encode(x))


with open("iwslt2017-en.txt", "w", encoding='UTF-8') as f_en, open("iwslt2017-de.txt", "w", encoding='UTF-8') as f_de:
    for it in dataset['train']:
        f_en.write(it["translation"]["en"] + "\n")
        f_de.write(it["translation"]["de"] + "\n")

spm.SentencePieceTrainer.train(
    input='iwslt2017-en.txt',
    model_prefix='en',
    model_type="bpe",
    )
spm.SentencePieceTrainer.train(
    input='iwslt2017-de.txt',
    model_prefix='de',
    model_type="bpe",
    )
en_sp = spm.SentencePieceProcessor(model_file='./de.model')
de_sp = spm.SentencePieceProcessor(model_file='./en.model')

train_df=dict_to_df(dataset['train'])
val_df=dict_to_df(dataset['validation'])
test_df=dict_to_df(dataset['test'])

tokenize(train_df,en_sp,de_sp)
tokenize(val_df,en_sp,de_sp)
tokenize(test_df,en_sp,de_sp)

train_df.to_pickle('./Dataset/train_df.pkl')
val_df.to_pickle('./Dataset/val_df.pkl')
test_df.to_pickle('./Dataset/test_df.pkl')

src_vocab_size=len(en_sp)
tgt_vocab_size=len(de_sp)