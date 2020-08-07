import cv2
import swag
import json
import tasti
import torch
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm.autonotebook import tqdm

# used for Mask-RCNN, Fast-RCNN, Detectron2, etc, ...
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# useful for converting object detector outputs into a List[Box]
class Box:
    def __init__(self, box, object_name, confidence):
        self.box = box
        self.xmin = box[0]
        self.ymin = box[1]
        self.xmax = box[2]
        self.ymax = box[3]
        self.object_name = object_name
        self.confidence = confidence
        
    def __str__(self):
        return f'Box(coords={list(self.box)}, {self.object_name}, {self.confidence})'
    
    def __repr__(self):
        return self.__str__()
    
# Video class that is comptatible with SWAG (https://github.com/stanford-futuredata/swag-python). 
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
        self.current_idx = 0
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
        if self.current_idx != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            self.current_idx = idx
        
    def read(self):
        _, frame = self.cap.read()
        frame = self.transform(frame)
        self.current_idx += 1
        return frame
    
    def __len__(self):
        return self.length if len(self.list_of_idxs) == 0 else len(self.list_of_idxs)
    
    def __getitem__(self, idx):
        if len(self.list_of_idxs) == 0:
            self.seek(idx)
            frame = self.read()
        else:
            frame = self.frames[idx]
        return frame   

# Label class for when all the Target DNN outputs are available
class Labels:
    def __init__(self, labels_fp, length):
        labels_fp = '/lfs/1/jtguibas/data/labels/jackson-town-square-2017-12-17.csv'
        df = pd.read_csv(labels_fp)
        df = df[df['object_name'].isin(['car'])]
        frame_to_rows = defaultdict(list)
        for row in df.itertuples():
            frame_to_rows[row.frame].append(row)
        labels = []
        for frame_idx in range(length):
            labels.append(frame_to_rows[frame_idx])
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx]

# assuming label1 and label2 are of only one object type, checks if they are redundant
def night_street_is_close_helper(self, label1, label2):
    if len(label1) != len(label2):
        return False
    counter = 0
    for obj1 in label1:
        xavg1 = (obj1.xmin + obj1.xmax) / 2.0
        yavg1 = (obj1.ymin + obj1.ymax) / 2.0
        coord1 = [xavg1, yavg1]
        expected_counter = counter + 1
        for obj2 in label2:
            xavg2 = (obj2.xmin + obj2.xmax) / 2.0
            yavg2 = (obj2.ymin + obj2.ymax) / 2.0
            coord2 = [xavg2, yavg2]
            if distance.euclidean(coord1, coord2) < self.threshold:
                counter += 1
                break
        if expected_counter != counter:
            break
    return len(label1) == counter

# creates disjoint sets where sets are specific to one object type, if all the corresponding sets are redundant, return True
def is_close(self, a, b):    
    objects = set()
    for obj in (label1 + label2):
        objects.add(obj.object_name)
    for current_obj in list(objects):
        label1_disjoint = [obj for obj in label1 if obj.object_name == current_obj]
        label2_disjoint = [obj for obj in label2 if obj.object_name == current_obj]
        is_redundant = self.redundant_helper(label1_disjoint, label2_disjoint)
        if not is_redundant:
            return False
    return True
    
# specific to jackson town square
def embedding_dnn_transform_fn(frame):
    xmin, xmax, ymin, ymax = 0, 1750, 540, 1080
    frame = frame[ymin:ymax, xmin:xmax]
    frame = cv2.resize(frame, (224, 224))
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

# specific to jackson town square
def target_dnn_transform_fn(frame):
    xmin, xmax, ymin, ymax = 0, 1750, 540, 1080
    frame = frame[ymin:ymax, xmin:xmax]
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame