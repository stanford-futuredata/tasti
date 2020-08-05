import torch
import tasti
import numpy as np
from tqdm.autonotebook import tqdm

class TargetDNNCache:
    def __init__(self, target_dnn, dataset, length):
        self.target_dnn = target_dnn
        self.dataset = dataset
        self.length = length
        self.cache = [None for i in range(self.length)]
        
    def __getitem__(self, idx):
        if self.cache[idx] == None:
            data = self.dataset[idx]
            out = self.target_dnn(data)
            self.cache[idx] = out
        return self.cache[idx]
    
class TargetDNNCacheArray(np.ndarray):
    def __new__(cls, target_dnn_cache, scoring_fn):
        self.target_dnn_cache = target_dnn_cache
        self.length = self.target_dnn_cache.length
        self.scoring_fn = scoring_fn
        arr = np.full(self.length, -1)
        obj = np.asarray(arr).view(cls)
        return obj
    
    def __getitem__(self, item):
        res = self.target_dnn_cache[item]
        outs = []
        if isinstance(item, (slice, int)):
            for thing in res:
                outs.append(self.scoring_fn(thing))
        else:
            outs = self.scoring_fn(res)
        super()[res] = outs
        return super().__getitem__(item)

class Index:
    def __init__(self, config):
        self.config = config
        
    def is_close(self, a, b):
        raise NotImplementedError
        
    def get_target_dnn_dataset(self):
        raise NotImplementedError
    
    def get_embedding_dnn_dataset(self):
        raise NotImplementedError
        
    def get_target_dnn(self):
        raise NotImplementedError
        
    def get_embedding_dnn(self):
        raise NotImplementedError
        
    def target_dnn_post_processing(self, target_dnn_output):
        return target_dnn_output
    
    def embedding_dnn_post_processing(self, embedding_dnn_output):
        return embedding_dnn_output
    
    def do_mining(self):
        if self.config.do_mining:
            model = self.get_embedding_dnn()
            model.eval()
            model.cuda()
            
            dataset = self.get_embedding_dnn_dataset()
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
            
            embeddings = []
            for batch in tqdm(dataloader, desc='FPF Mining'):
                output = model(batch)
                embeddings.append(output)    
            embeddings = torch.cat(embeddings, dim=0)
            
            bucketter = tasti.bucketter.FPFBucketter(self.config.nb_train)
            reps, _, _ = bucketter.bucket(self.embeddings)
            self.training_idxs = reps
        else:
            self.training_idxs = self.rand.choice(
                    len(self.get_embedding_dnn_dataset()),
                    size=self.config.nb_train,
                    replace=False
            )
            
    def do_training(self):
        model = self.get_embedding_dnn()
        if self.config.do_training:
            model.train()
            model.cuda()
            
            triplet_dataset = self.get_triplet_dataset()
            loss_fn = TripletLoss(self.config.train_margin)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train_lr)
            dataloader = torch.utils.data.DataLoader(
                    triplet_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True
            )
            
            for epoch in tqdm.tqdm(range(self.config.epochs), desc='Epoch'):
                for anchor, positive, negative in tqdm.tqdm(dataloader, desc='Step'):
                    anchor = anchor.cuda(non_blocking=True)
                    positive = positive.cuda(non_blocking=True)
                    negative = negative.cuda(non_blocking=True)

                    e_a = model(anchor)
                    e_p = model(positive)
                    e_n = model(negative)

                    optimizer.zero_grad()
                    loss = loss_fn(e_a, e_p, e_n)
                    loss.backward()
                    optimizer.step()
                    
        torch.save(model.state_dict(), 'model.pth')
        self.embedding_dnn = model
        
    def do_bucketting(self):
        bucketter = tasti.bucketter.FPFBucketter(self.config.nb_buckets))
        self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.k)
            
            
        
    
        
    