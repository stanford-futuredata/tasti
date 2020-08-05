import cv2
import swag
import json
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm

class Video(torch.utils.data.Dataset):
    def __init__(self, video_fp, list_of_idxs=[]):
        self.video_fp = video_fp
        self.list_of_idxs = []
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
#         xmin, xmax, ymin, ymax = 0, 1750, 540, 1080
#         frame = frame[ymin:ymax, xmin:xmax]
#         frame = cv2.resize(frame, (224, 224))
        frame = transforms.functional.to_tensor(frame)
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