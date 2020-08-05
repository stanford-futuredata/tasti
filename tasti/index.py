import torch
import tasti
from tqdm.autonotebook import tqdm

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
            
            
        
    
        
    