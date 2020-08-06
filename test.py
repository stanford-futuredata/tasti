import os
import cv2
import tasti
import torch
import pickle
import torchvision
import numpy as np

os.environ['TORCH_HOME'] = '/lfs/1/jtguibas/models'
os.environ['FVCORE_CACHE'] = '/lfs/1/jtguibas/models'

class IndexConfig:
    def __init__(self):
        self.do_mining = True
        self.do_training = True
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 7000
        self.nb_training_its = 12000
        
config = IndexConfig()

class MyIndex(tasti.Index):
    def is_close(self, a, b):
        return len(a) == len(b)
    
    def get_target_dnn_dataset(self):
        video = tasti.Video(
            video_fp='/lfs/1/jtguibas/data/2017-12-17',
            transform_fn=tasti.target_dnn_transform_fn
        )
        return video
    
    def get_embedding_dnn_dataset(self):
        video = tasti.Video(
            video_fp='/lfs/1/jtguibas/data/2017-12-17',
            transform_fn=tasti.embedding_dnn_transform_fn
        )
        return video
    
    def target_dnn_callback(self, result, threshold=0.95):
        
        boxes = result[0]['boxes'].detach().cpu().numpy()
        confidences = result[0]['scores'].detach().cpu().numpy()
        object_names = result[0]['labels'].detach().cpu().numpy()
        object_names = np.array([tasti.COCO_INSTANCE_CATEGORY_NAMES[l] for l in object_names])
        criteria = confidences > threshold

        boxes = boxes[criteria]
        object_names = object_names[criteria]
        confidences = confidences[criteria]
        
        return [tasti.Box(x[0], x[1], x[2]) for x in zip(boxes, object_names, confidences)]
        
    def get_target_dnn(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        return model
        
    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        return model
    
index = MyIndex(config)
index.do_mining()
index.do_training()
index.do_infer()
index.do_bucketting()
pickle.dump(index, open('index.pkl', 'wb'))

index = pickle.load(open('index.pkl', 'rb'))
query = tasti.Query(index)
query.execute()