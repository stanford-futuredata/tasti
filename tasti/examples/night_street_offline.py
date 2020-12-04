'''
This code allows you to reproduce the results in the paper corresponding to the "night-street" dataset.
The term 'offline' refers to the fact that all the target dnn outputs have already been computed.
If you like to run the 'online' version (target dnn runs in realtime), take a look at "night_street_online.py". 
Look at the README.md file for information about how to get the data to run this code.
'''
import os
import cv2
import swag
import json
import tasti
import torch
import pandas as pd
import numpy as np
import torchvision
from scipy.spatial import distance
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm.autonotebook import tqdm
from blazeit.aggregation.samplers import ControlCovariateSampler

# Feel free to change this!
ROOT_DATA_DIR = '/lfs/1/jtguibas/data'

'''
VideoDataset allows you to access frames of a given video.
'''
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_fp, list_of_idxs=[], transform_fn=lambda x: x):
        self.video_fp = video_fp
        self.list_of_idxs = []
        self.transform_fn = transform_fn
        self.cap = swag.VideoCapture(self.video_fp)
        self.video_metadata = json.load(open(self.video_fp + '.json', 'r'))
        self.cum_frames = np.array(self.video_metadata['cum_frames'])
        self.cum_frames = np.insert(self.cum_frames, 0, 0)
        self.length = self.cum_frames[-1]
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

'''
LabelDataset loads the target dnn .csv files and allows you to access the target dnn outputs of given frames.
'''
class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, labels_fp, length):
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

'''
Preprocessing function of a frame before it is passed to the Embedding DNN.
'''
def night_street_embedding_dnn_transform_fn(frame):
    xmin, xmax, ymin, ymax = 0, 1750, 540, 1080
    frame = frame[ymin:ymax, xmin:xmax]
    frame = cv2.resize(frame, (224, 224))
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

def night_street_target_dnn_transform_fn(frame):
    xmin, xmax, ymin, ymax = 0, 1750, 540, 1080
    frame = frame[ymin:ymax, xmin:xmax]
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

'''
Defines our notion of 'closeness' as described in the paper for two labels for only one object type.
'''
def night_street_is_close_helper(label1, label2):
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
            if distance.euclidean(coord1, coord2) < 100:
                counter += 1
                break
        if expected_counter != counter:
            break
    return len(label1) == counter
        
class NightStreetOfflineIndex(tasti.Index):
    def get_target_dnn(self):
        '''
        In this case, because we are running the target dnn offline, so we just return the identity.
        '''
        model = torch.nn.Identity()
        return model
        
    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        return model
    
    def get_pretrained_embedding_dnn(self):
        '''
        Note that the pretrained embedding dnn sometime differs from the embedding dnn.
        '''
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model
    
    def get_target_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-12-14')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-12-17')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=night_street_target_dnn_transform_fn
        )
        return video
    
    def get_embedding_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-12-14')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-12-17')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=night_street_embedding_dnn_transform_fn
        )
        return video
    
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        if train_or_test == 'train':
            labels_fp = os.path.join(ROOT_DATA_DIR, 'jackson-town-square-2017-12-14.csv')
        else:
            labels_fp = os.path.join(ROOT_DATA_DIR, 'jackson-town-square-2017-12-17.csv')
        labels = LabelDataset(
            labels_fp=labels_fp,
            length=len(target_dnn_cache)
        )
        return labels
    
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

class NightStreetAggregateQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
class NightStreetLimitQuery(tasti.LimitQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
class NightStreetSUPGPrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0
    
class NightStreetSUPGRecallQuery(tasti.SUPGRecallQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0
    
class NightStreetLHSPrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False
        return proc_boxes(target_dnn_output)
    
class NightStreetLHSRecallQuery(tasti.SUPGRecallQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False
        return proc_boxes(target_dnn_output)
    
class NightStreetAveragePositionAggregateQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            avg = 0.
            if len(boxes) == 0:
                return 0.
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                avg += x / 1750
            return avg / len(boxes)
        return proc_boxes(target_dnn_output)
    
class NightStreetOfflineConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = True
        self.do_training = True
        self.do_infer = True
        self.do_bucketting = True
        
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 7000
        self.nb_training_its = 12000
    
if __name__ == '__main__':
    config = NightStreetOfflineConfig()
    index = NightStreetOfflineIndex(config)
    index.init()

    query = NightStreetAggregateQuery(index)
    query.execute_metrics(err_tol=0.01, confidence=0.05)
    
    query = NightStreetAveragePositionAggregateQuery(index)
    query.execute_metrics(err_tol=0.005, confidence=0.05)

    query = NightStreetLimitQuery(index)
    query.execute_metrics(want_to_find=5, nb_to_find=10)

    query = NightStreetSUPGPrecisionQuery(index)
    query.execute_metrics(10000)

    query = NightStreetSUPGRecallQuery(index)
    query.execute_metrics(10000)

    query = NightStreetLHSPrecisionQuery(index)
    query.execute_metrics(10000)

    query = NightStreetLHSRecallQuery(index)
    query.execute_metrics(10000)    
