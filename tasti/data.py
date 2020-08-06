import cv2
import swag
import json
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm

class TripletDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset,
            target_dnn_outputs,
            list_of_idxs,
            is_close_fn,
            length=1000
    ):
        self.dataset = dataset
        self.target_dnn_outputs = target_dnn_outputs
        self.list_of_idxs = list_of_idxs
        self.is_close_fn = is_close_fn
        self.length = length

        self.buckets = []
        for idx in tqdm(self.list_of_idxs, desc="Triplet Dataset Init"):
            label = self.target_dnn_outputs[idx]
            found = False
            for bucket in self.buckets:
                rep_idx = bucket[0]
                rep = self.target_dnn_outputs[rep_idx]
                if self.is_close_fn(label, rep):
                    bucket.append(idx)
                    found = True
                    break
            if not found:
                self.buckets.append([idx])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rand = np.random.RandomState(seed=idx)
        rand.randint(0, 100, size=10)
        
        def get_triplet_helper():
            anchor_bucket_idx = rand.randint(0, len(self.buckets))
            anchor_bucket = self.buckets[anchor_bucket_idx]
            negative_bucket_idx = rand.choice(
                    [idx for idx in range(len(self.buckets)) if idx != anchor_bucket_idx]
            )
            negative_bucket = self.buckets[negative_bucket_idx]

            anchor_idx = rand.choice(anchor_bucket)
            positive_idx = rand.choice(anchor_bucket)
            negative_idx = rand.choice(negative_bucket)

            return anchor_idx, positive_idx, negative_idx

        anchor_idx, positive_idx, negative_idx = get_triplet_helper()
        for i in range(200):
            if abs(anchor_idx - positive_idx) > 30:
                break
            else:
                anchor_idx, positive_idx, negative_idx = get_triplet_helper()

        anchor = self.dataset[anchor_idx]
        positive = self.dataset[positive_idx]
        negative = self.dataset[negative_idx]
        
        return anchor, positive, negative
    
    
# =========================================================================================

class Video(torch.utils.data.Dataset):
    def __init__(self, video_fp, list_of_idxs=[], transform_fn=lambda x: x):
        self.video_fp = video_fp
        self.list_of_idxs = []
        self.transform_fn = transform_fn
        self.video_metadata = json.load(open(self.video_fp + '.json', 'r'))
        self.cum_frames = np.array(self.video_metadata['cum_frames'])
        self.cum_frames = np.insert(self.cum_frames, 0, 0)
        self.length = self.cum_frames[-1]
        self.cap = swag.VideoCapture(self.video_fp)
        self.init()
        
    def init(self):
        if len(self.list_of_idxs) == 0:
            self.frames = None
        else:
            self.frames = []
            for idx in tqdm(self.list_of_idxs, desc="Video"):
                self.seek(idx)
                frame = self.read()
                self.frames.append(frame)
            
    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform_fn(frame)
        return frame

    def seek(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        
    def read(self):
        _, frame = self.cap.read()
        frame = self.transform(frame)
        return frame
    
    def __len__(self):
        return self.length if len(self.list_of_idxs) == 0 else len(self.list_of_idxs)
    
    def __getitem__(self, idx):
        if len(self.list_of_idxs) == 0:
            frame = self.read()
        else:
            frame = self.frames[idx]
        return frame   
    
def embedding_dnn_transform_fn(frame):
    xmin, xmax, ymin, ymax = 0, 1750, 540, 1080
    frame = frame[ymin:ymax, xmin:xmax]
    frame = cv2.resize(frame, (224, 224))
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

def target_dnn_transform_fn(frame):
    xmin, xmax, ymin, ymax = 0, 1750, 540, 1080
    frame = frame[ymin:ymax, xmin:xmax]
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame