class IndexConfig:
    def __init__(self):
        self.do_mining = False
        self.do_training = False
        self.do_infer = False
        self.do_bucketting = False
        
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 7000
        self.nb_training_its = 12000