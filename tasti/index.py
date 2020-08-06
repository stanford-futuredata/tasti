import torch
import torchvision
import tasti
import numpy as np
from tqdm.autonotebook import tqdm

class Index:
    def __init__(self, config):
        self.config = config
        
    def get_is_close_fn(self, a, b):
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
        return target_dnn_output

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
            for batch in tqdm(dataloader, desc='FPF Mining'):
                batch = batch.cuda()
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)  
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()
            
            bucketter = tasti.bucketters.FPFBucketter(self.config.nb_train)
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
            
            dataset = self.get_target_dnn_dataset()
            self.target_dnn_outputs = [None for i in range(10000)]
            
            for idx in tqdm(self.training_idxs, desc='Target DNN Invocations'):
                data = dataset[idx].unsqueeze(0).cuda() # is .unsqueeze bad?
                with torch.no_grad():
                    out = model(data) 
                    try:
                        out = out.cpu()
                    except:
                        pass
                    out = self.target_dnn_callback(out)
                self.target_dnn_outputs[idx] = out
            
            del dataset
            del model
            
            dataset = self.get_embedding_dnn_dataset()
            triplet_dataset = tasti.data.TripletDataset(
                dataset=dataset,
                target_dnn_outputs=self.target_dnn_outputs,
                list_of_idxs=self.training_idxs,
                is_close_fn=self.is_close
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
            
            for anchor, positive, negative in tqdm(dataloader, desc='Step'):
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
                
            torch.save(model.state_dict(), 'model.pt')
            self.embedding_dnn_trained = model
            
            del dataset
            del triplet_dataset
            del dataloader
            
    def do_infer(self):
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
        
    def do_bucketting(self):
        bucketter = tasti.bucketters.FPFBucketter(self.config.nb_buckets)
        self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k)