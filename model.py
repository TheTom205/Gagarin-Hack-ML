import numpy as np
import pandas as pd
import torch
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertForMaskedLM,BertTokenizer, pipeline

from transformers import AutoTokenizer, AutoModel
import torch

# model=BertForMaskedLM.from_pretrained('sberbank-ai/ruBert-base')
generator = pipeline("text2text-generation", model="sberbank-ai/ruT5-base")

tokenizer = AutoTokenizer.from_pretrained('uaritm/multilingual_en_uk_pl_ru')
model = AutoModel.from_pretrained('uaritm/multilingual_en_uk_pl_ru')
df=pd.read_csv('/home/thetom205/MISIS-gagarin/2024-04-13 Gagarin Hack Profiles (4).csv')
df_renamed=df.rename(columns={'Напишите о себе (ваш пол, чем увлекаетесь, на каком языке пишете код, в каких областях хотели бы развиваться, какие ваши сильные стороны)':'description','На какой IT курс ходите или хотели бы ходить? / ML':'ML','На какой IT курс ходите или хотели бы ходить? / Docker':'Docker','На какой IT курс ходите или хотели бы ходить? / Backend':'Backend','На какой IT курс ходите или хотели бы ходить? / Frontend':'Frontend','На какой IT курс ходите или хотели бы ходить? / Дизайн':'Дизайн','На какой IT курс ходите или хотели бы ходить? / Робототехника':'Робототехника','На какой IT курс ходите или хотели бы ходить? / Swift':'Swift','На какой IT курс ходите или хотели бы ходить? / Go':'Go'})
names=['ML','Docker','Backend','Frontend','Дизайн','Робототехника','Swift','Go']
for i in names:
    df_renamed[i]=df_renamed[i].replace(i,1)

for i in names:
    df_renamed[i]=df_renamed[i].fillna(0)

df_renamed_new=df_renamed[1:]

df_renamed_new.reset_index(drop=True, inplace=True)
description_usrs=df_renamed['description']

class CustomDataset(Dataset):
    
    def __init__(self, X):
        self.text = X

    def tokenize(self, text):
        return tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=150)

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, index):
        output = self.text[index]
        output = self.tokenize(output)
        return {k: v.reshape(-1) for k, v in output.items()}


#eval_ds = CustomDataset(description_usrs)
#eval_dataloader = DataLoader(eval_ds, batch_size=10)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

embeddings=pd.read_csv('/home/thetom205/MISIS-gagarin/embeddings2.csv')

embeddings=embeddings.to_numpy()
embeddings

def find_courses(user_text):
    answers = [user_text]
    eval_ds_usr = CustomDataset(np.array(answers))
    eval_dataloader_usr = DataLoader(eval_ds_usr, batch_size=10)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    embeddings_usr = torch.Tensor().to(device)

    with torch.no_grad():
        for n_batch, batch in enumerate(eval_dataloader_usr):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings_usr = torch.cat([embeddings_usr, mean_pooling(outputs, batch['attention_mask'])])
        embeddings_usr = embeddings_usr.cpu().numpy()
    cosine_similarities = cosine_similarity( embeddings_usr, embeddings)
    df_renamed_new['cos']=pd.Series(cosine_similarities[0])
    X=embeddings
    y=df_renamed_new[names]
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=5,algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    neigh.fit(X, y)
    res=neigh.predict_proba(embeddings_usr)
    data = [['ML',res[0][0][1]],['Docker',res[1][0][1]],['Backend',res[2][0][1]],['Frontend',res[3][0][1]],['Дизайн',res[4][0][1]],['Робототехника',res[5][0][1]],['Swift',res[6][0][1]],['Go',res[7][0][1]]]
 
    df_res = pd.DataFrame(data, columns=['Course','result'])
    sort_df=df_renamed_new.sort_values(by=['cos'],ascending=False).head()
    
    return df_res

def find_friends(user_text):
    answers = [user_text]
    eval_ds_usr = CustomDataset(np.array(answers))
    eval_dataloader_usr = DataLoader(eval_ds_usr, batch_size=10)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    embeddings_usr = torch.Tensor().to(device)

    with torch.no_grad():
        for n_batch, batch in enumerate(eval_dataloader_usr):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings_usr = torch.cat([embeddings_usr, mean_pooling(outputs, batch['attention_mask'])])
        embeddings_usr = embeddings_usr.cpu().numpy()
    cosine_similarities = cosine_similarity( embeddings_usr, embeddings)
    df_renamed_new['cos']=pd.Series(cosine_similarities[0])
    sort_df=df_renamed_new.sort_values(by=['cos'],ascending=False).head()
    
    return sort_df['ID'].astype('object')