import torch
import torchvision
import tasti
import numpy as np
from tqdm.autonotebook import tqdm

class Index:
    def __init__(self, config):
        self.config = config
        self.target_dnn_cache = tasti.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(),
            self.target_dnn_callback
        )
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache)
        self.rand = np.random.RandomState(seed=1)
        
    def override_target_dnn_cache(self, target_dnn_cache):
        return target_dnn_cache
        
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
        
    def target_dnn_callback(self, target_dnn_output):
        return len(target_dnn_output)

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
            )
            
            embeddings = []
            for batch in tqdm(dataloader, desc='Embedding DNN'):
                batch = batch.cuda()
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)  
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()
            
            bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_train)
            reps, _, _ = bucketter.bucket(embeddings, self.config.max_k)
            self.training_idxs = reps
        else:
            self.training_idxs = self.rand.choice(
                    len(self.get_embedding_dnn_dataset()),
                    size=self.config.nb_train,
                    replace=False
            )
            
    def do_training(self):
        if self.config.do_training:
            model = self.get_target_dnn()
            model.eval()
            model.cuda()
            
            for idx in tqdm(self.training_idxs, desc='Target DNN'):
                self.target_dnn_cache[idx]
            
            dataset = self.get_embedding_dnn_dataset()
            triplet_dataset = tasti.data.TripletDataset(
                dataset=dataset,
                target_dnn_cache=self.target_dnn_cache,
                list_of_idxs=self.training_idxs,
                is_close_fn=self.is_close,
                length=self.config.nb_training_its
            )
            dataloader = torch.utils.data.DataLoader(
                triplet_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            
            model = self.get_embedding_dnn()
            model.train()
            model.cuda()
            loss_fn = tasti.TripletLoss(self.config.train_margin)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train_lr)
            
            for anchor, positive, negative in tqdm(dataloader, desc='Training Step'):
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
                
            torch.save(model.state_dict(), './cache/model.pt')
            self.embedding_dnn_trained = model
            
    def do_infer(self):
        if self.config.do_infer:
            model = self.embedding_dnn_trained
            model.eval()
            model.cuda()
            dataset = self.get_embedding_dnn_dataset()
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            embeddings = []
            for batch in tqdm(dataloader, desc='Inference'):
                batch = batch.cuda()
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)  
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()

            np.save('embeddings.npy', embeddings)
            self.embeddings = embeddings
        else:
            self.embeddings = np.load('./cache/embeddings.npy')
        
    def do_bucketting(self):
        if self.config.do_bucketting:
            bucketter = tasti.bucketters.FPFBucketter(self.config.nb_buckets)
            self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k)
            np.save('./cache/reps.npy', self.reps)
            np.save('./cache/topk_reps.npy', self.topk_reps)
            np.save('./cache/topk_dists.npy', self.topk_dists)
        else:
            self.reps = np.load('./cache/reps.npy')
            self.topk_reps = np.load('./cache/topk_reps.npy')
            self.topk_dists = np.load('./cache/topk_dists.npy')
            
    def init(self):
        self.do_mining()
        self.do_training()
        self.do_infer()
        self.do_bucketting()
        
        for rep in tqdm(self.reps, desc='Target DNN Invocations'):
            self.target_dnn_cache[rep]