import tasti
import jsonlines
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from torchnlp.word_to_vector import FastText

class WikiSQLDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl):
        self.mode = 'input'
        data = []
        with jsonlines.open(jsonl) as reader:
            for obj in reader:
                sql = obj['sql']
                label = (sql['agg'], len(sql['conds']))
                text = obj['question'].strip().lower()
                text = text.replace('?', '') # TODO: ??
                data.append((text, sql['agg'], len(sql['conds'])))
        self.df = pd.DataFrame(data, columns=['text', 'agg', 'conds'])
        self.vectors = FastText(cache='./cache/vectors.pth')

    def __len__(self):
        return len(self.df)

    def embed(self, text):
        words = text.split(' ') # FIXME
        emb = self.vectors[words[0]]
        for word in words[1:]:
            emb += self.vectors[word]
        emb /= len(words)
        return emb
    
    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']
        agg = self.df.loc[idx, 'agg']
        conds = self.df.loc[idx, 'conds']
        if self.mode == 'input':
            return self.embed(text)
        else:
            return agg, conds
        
class Embedder(nn.Module):
    def __init__(self, nb_out=128):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(300, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, nb_out)
        )

    def forward(self, x):
        return self.mlp(x)

class WikiSQLOfflineIndex(tasti.Index):
    def get_target_dnn(self):
        model = torch.nn.Identity()
        return model
        
    def get_embedding_dnn(self):
        model = Embedder()
        return model
    
    def get_target_dnn_dataset(self):
        sql_dataset = WikiSQLDataset('/lfs/1/jtguibas/text/data/train.jsonl')
        sql_dataset.mode = 'input'
        return sql_dataset
    
    def get_embedding_dnn_dataset(self):
        sql_dataset = WikiSQLDataset('/lfs/1/jtguibas/text/data/train.jsonl')
        sql_dataset.mode = 'input'
        return sql_dataset
    
    def override_target_dnn_cache(self, target_dnn_cache):
        sql_dataset = WikiSQLDataset('/lfs/1/jtguibas/text/data/train.jsonl')
        sql_dataset.mode = 'output'
        return sql_dataset
    
    def is_close(self, label1, label2):
        return label1 == label2
    
class WikiSQLAggregateQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        return target_dnn_output[1]
    
class WikiSQLSUPGPrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        return 1.0 if target_dnn_output[0] == 0 else 0.0
    
class WikiSQLOfflineConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = True
        self.do_training = True
        self.do_infer = True
        self.do_bucketting = True
        self.nb_train = 500
        self.nb_buckets = 500
        self.batch_size = 1
        
if __name__ == '__main__':
    config = WikiSQLOfflineConfig()
    index = WikiSQLOfflineIndex(config)
    index.init()
    
    query = WikiSQLAggregateQuery(index)
    result = query.execute()
    print(result)
    
    query = WikiSQLSUPGPrecisionQuery(index)
    result = query.execute()
    print(result)