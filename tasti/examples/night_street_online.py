import os
import cv2
import swag
import json
import tasti
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from scipy.spatial import distance
from collections import defaultdict
import torchvision.transforms as transforms
from tasti.examples.night_street_offline import VideoDataset
from tasti.examples.night_street_offline import night_street_is_close_helper
from tasti.examples.night_street_offline import night_street_embedding_dnn_transform_fn
from tasti.examples.night_street_offline import night_street_target_dnn_transform_fn
from tasti.examples.night_street_offline import NightStreetAggregateQuery
from tasti.examples.night_street_offline import NightStreetSUPGPrecisionQuery
from tasti.examples.night_street_offline import NightStreetSUPGRecallQuery

os.environ['TORCH_HOME'] = '/lfs/1/jtguibas/models'
os.environ['FVCORE_CACHE'] = '/lfs/1/jtguibas/models'

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
        return f'Box({self.xmin},{self.ymin},{self.xmax},{self.ymax},{self.object_name},{self.confidence})'
    
    def __repr__(self):
        return self.__str__()
    
class NightStreetOnlineIndex(tasti.Index):
    def get_target_dnn(self):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)
        return model
        
    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        return model
    
    def get_target_dnn_dataset(self):
        video = VideoDataset(
            video_fp='/lfs/1/jtguibas/data/2017-12-17',
            transform_fn=night_street_target_dnn_transform_fn
        )
        return video
    
    def get_embedding_dnn_dataset(self):
        video = VideoDataset(
            video_fp='/lfs/1/jtguibas/data/2017-12-17',
            transform_fn=night_street_embedding_dnn_transform_fn
        )
        return video
    
    def target_dnn_callback(self, target_dnn_output):
        boxes = target_dnn_output[0]['boxes'].detach().cpu().numpy()
        confidences = target_dnn_output[0]['scores'].detach().cpu().numpy()
        object_ids = target_dnn_output[0]['labels'].detach().cpu().numpy()
        label = []
        for i in range(len(boxes)):
            object_name = COCO_INSTANCE_CATEGORY_NAMES[object_ids[i]]
            if confidences[i] > 0.95 and object_name in ['car']:
                box = Box(boxes[i], object_ids[i], confidences[i])
                label.append(box)
        return label
        
    def is_close(self, label1, label2):
        objects = set()
        for obj in (label1 + label2):
            objects.add(obj.object_name)
        for current_obj in list(objects):
            label1_disjoint = [obj for obj in label1 if obj.object_name == current_obj]
            label2_disjoint = [obj for obj in label2 if obj.object_name == current_obj]
            is_redundant = night_street_is_close_helper(label1_disjoint, label2_disjoint)
            if not is_redundant:
                return False
        return True

class NightStreetOnlineConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = False
        self.do_training = False
        self.do_infer = False
        self.do_bucketting = True
        
        self.batch_size = 16
        self.nb_train = 500
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 500
        self.nb_training_its = 1000
        
if __name__ == '__main__':
    config = NightStreetOfflineConfig()
    index = NightStreetOfflineIndex(config)
    index.init()

    query = NightStreetAggregateQuery(index)
    query.execute()

    query = NightStreetLimitQuery(index)
    query.execute(5)

    query = NightStreetSUPGPrecisionQuery(index)
    query.execute()

    query = NightStreetSUPGRecallQuery(index)
    query.execute()

    query = NightStreetLHSPrecisionQuery(index)
    query.execute()

    query = NightStreetLHSRecallQuery(index)
    query.execute()

    query = NightStreetAveragePositionAggregateQuery(index)
    query.execute()