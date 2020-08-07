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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']
        agg = self.df.loc[idx, 'agg']
        conds = self.df.loc[idx, 'conds']
        if self.mode == 'input':
            return text
        else:
            return agg, conds

class FastTextEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vectors = FastText(cache='./vectors.pth')
        
    def forward(self, sentences):
        embs = []
        for x in sentences:
            words = x.split(' ')
            emb = self.vectors[words[0]]
            for word in words[1:]:
                emb += self.vectors[word]
            emb /= len(words)
            embs.append(emb.reshape(1, -1))
        embs = torch.cat(embs, dim=0)
        return embs

class WikiSQLOfflineIndex(tasti.Index):
    def get_target_dnn(self):
        model = torch.nn.Identity()
        return model
        
    def get_embedding_dnn(self):
        model = FastTextEmbedder()
        return model
    
    def get_target_dnn_dataset(self):
        sql_dataset = WikiSQLDataset('../text/data/dev.jsonl')
        sql_dataset.mode = 'input'
        return sql_dataset
    
    def get_embedding_dnn_dataset(self):
        sql_dataset = WikiSQLDataset('../text/data/dev.jsonl')
        sql_dataset.mode = 'input'
        return sql_dataset
    
    def override_target_dnn_cache(self, target_dnn_cache):
        sql_dataset = WikiSQLDataset('../text/data/dev.jsonl')
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
        self.do_mining = False
        self.do_training = False
        self.do_infer = True
        self.nb_buckets = 500
        self.batch_size = 1
        
        
if __name__ == '__main__':
    index = WikiSQLOfflineIndex(config)
    index.init()

    query = NightStreetAggregateQuery(index)
    query.execute()

    query = NightStreetSUPGPrecisionQuery(index)
    query.execute()