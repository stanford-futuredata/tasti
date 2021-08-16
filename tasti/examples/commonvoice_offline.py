"""
To run this file, you will need to install https://github.com/qiuqiangkong/audioset_tagging_cnn
and download the pretrained ResNet22 and Cnn10 audio models as well as download the CommonVoice
audio dataset https://commonvoice.mozilla.org/en/datasets.
"""
import os
import sys
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import pandas as pd
from tqdm.auto import tqdm

import audioset_tagging_cnn
from audioset_tagging_cnn.utils.utilities import create_folder, get_filename
from audioset_tagging_cnn.pytorch.models import *
from audioset_tagging_cnn.pytorch.pytorch_utils import move_data_to_device
import audioset_tagging_cnn.utils.config as config

import tasti
import torch
import supg.datasource as datasource
import pandas as pd
import torchvision
import numpy as np
from supg.sampler import ImportanceSampler
from supg.selector import ApproxQuery
from supg.selector import RecallSelector, ImportancePrecisionTwoStageSelector

import torchaudio
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


sample_rate = 32000
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
model_type_resnet22 = "ResNet22"
checkpoint_path_resnet22 = "audioset_tagging_cnn/ResNet22_mAP=0.430.pth"
model_type_cnn10 = "Cnn10"
checkpoint_path_cnn10 = "audioset_tagging_cnn/Cnn10_mAP=0.380.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classes_num = config.classes_num
labels = config.labels


ResNet22 = eval(model_type_resnet22)
resnet22 = ResNet22(sample_rate=sample_rate, window_size=window_size, 
    hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
    classes_num=classes_num)

checkpoint_resnet22 = torch.load(checkpoint_path_resnet22, map_location=device)
resnet22.load_state_dict(checkpoint_resnet22['model'])


Cnn10 = eval(model_type_cnn10)
cnn10 = Cnn10(sample_rate=sample_rate, window_size=window_size, 
    hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
    classes_num=classes_num)

checkpoint_cnn10 = torch.load(checkpoint_path_cnn10, map_location=device)
cnn10.load_state_dict(checkpoint_cnn10['model'])


if 'cuda' in str(device):
    resnet22.to(device)
    resnet22 = torch.nn.DataParallel(resnet22)
    cnn10.to(device)
    cnn10 = torch.nn.DataParallel(cnn10)
else:
    print('Using CPU.')


class ChangeSampleRate(nn.Module):
    def __init__(self, input_rate: int, output_rate: int):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1)
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        indices = (torch.arange(new_length) * (self.input_rate / self.output_rate))
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1. - indices.fmod(1.)).unsqueeze(0) + round_up * indices.fmod(1.).unsqueeze(0)
        return output


class CommonVoiceDataset(torch.utils.data.Dataset):
    def __init__(self, version="both"):
        self.version = version
        self.df = pd.read_csv("commonvoice_train_filtered.csv")
        self.df["age"][self.df["age"] <= 15] = 0
        self.df["age"][(15 < self.df["age"]) & (self.df["age"] <= 60)] = 1
        self.df["age"][60 < self.df["age"]] = 2
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = "/lfs/1/jtguibas/datasets/CommonVoice/cv-corpus-6.1-2020-12-11/en/clips/" + row["path"]
        waveform_raw, input_sr = torchaudio.load(fp)
        waveform_raw = ChangeSampleRate(input_sr, sample_rate)(waveform_raw).numpy().reshape(-1)
        waveform = np.zeros(sample_rate*3)
        waveform_raw = waveform_raw[:sample_rate*3]
        waveform[:waveform_raw.shape[0]] = waveform_raw
        if self.version == "both":
            return torch.Tensor(waveform), torch.Tensor([row.age, row.gender])
        elif self.version == "input":
            return torch.Tensor(waveform)
        elif self.version == "output":
            return torch.Tensor([row.age, row.gender])
        elif self.version == "age":
            return torch.Tensor(waveform), row.age
        elif self.version == "gender":
            return torch.Tensor(waveform), row.gender


class CommonVoiceIndex(tasti.Index):
    def is_close(self, a, b):
        return (a[0] == b[0]) and (a[1] == b[1])

    def get_target_dnn_dataset(self, train_or_test='train'):
        return CommonVoiceDataset(version="input")

    def get_embedding_dnn_dataset(self, train_or_test='train'):
        return self.get_target_dnn_dataset(train_or_test)
    
    def get_pretrained_embedding_dnn(self, out_size=None):
        ResNet22 = eval(model_type_resnet22)
        resnet22 = ResNet22(sample_rate=sample_rate, window_size=window_size, 
            hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
            classes_num=classes_num)
        checkpoint_resnet22 = torch.load(checkpoint_path_resnet22, map_location=device)
        resnet22.load_state_dict(checkpoint_resnet22['model'])
        if out_size != None:
            resnet22.fc_audioset.out_features = out_size
        resnet22.to(device)
        resnet22 = torch.nn.DataParallel(resnet22)
        return resnet22
    
    def get_embedding_dnn(self):
        return self.get_pretrained_embedding_dnn(out_size=128)
    
    def get_target_dnn(self):
        model = torch.nn.Identity()
        return model

    def override_target_dnn_cache(self, target_dnn_cache, train_or_test='train'):
        return CommonVoiceDataset(version="output")


class CommonVoiceOfflineConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = True
        self.do_training = True
        self.do_infer = True
        self.do_bucketting = True
        
        self.batch_size = 64
        self.nb_train = 500
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 500
        self.nb_training_its = 12000


TRIALS = 10

for TRIAL in range(TRIALS):
    print("TRIAL:", TRIAL)
    config = CommonVoiceOfflineConfig()
    config.seed = TRIAL
    print("NB_TRAIN", config.nb_train)
    print("NB_BUCKETS", config.nb_buckets)
    index = CommonVoiceIndex(config)
    index.init()


    class AgeAggregateQuery(tasti.AggregateQuery):
        def score(self, target_dnn_output):
            return target_dnn_output["age"].to_numpy()

    class GenderAggregateQuery(tasti.AggregateQuery):
        def score(self, target_dnn_output):
            return target_dnn_output["gender"].to_numpy()

    class AgeLimitQuery(tasti.LimitQuery):
        def score(self, target_dnn_output):
            return target_dnn_output["age"].to_numpy()

    class AgePrecisionQuery(tasti.SUPGPrecisionQuery):
        def score(self, target_dnn_output):
            return (target_dnn_output["age"].to_numpy() > 0).astype(int)

    class AgeRecallQuery(tasti.SUPGRecallQuery):
        def score(self, target_dnn_output):
            return (target_dnn_output["age"].to_numpy() > 0).astype(int)

    class GenderPrecisionQuery(tasti.SUPGPrecisionQuery):
        def score(self, target_dnn_output):
            return (target_dnn_output["gender"].to_numpy() > 0).astype(int)

    class GenderRecallQuery(tasti.SUPGRecallQuery):
        def score(self, target_dnn_output):
            return (target_dnn_output["gender"].to_numpy() > 0).astype(int)


    query = AgeAggregateQuery(index)
    query.df = True
    query.execute_metrics(err_tol=0.01, confidence=0.05, trials=1)

    query = GenderAggregateQuery(index)
    query.df = True
    query.execute_metrics(err_tol=0.01, confidence=0.05, trials=1)

    query = AgeLimitQuery(index)
    query.df = True
    query.execute_metrics(want_to_find=2, nb_to_find=10, GAP=0)

    query = AgePrecisionQuery(index)
    query.df = True
    query.execute_metrics(10000)
    
    query = GenderPrecisionQuery(index)
    query.df = True
    query.execute_metrics(10000)

    query = AgeRecallQuery(index)
    query.df = True
    query.execute_metrics(10000)

    query = GenderRecallQuery(index)
    query.df = True
    query.execute_metrics(10000)


    class PL(pl.LightningModule):
        def __init__(self, out_size):
            super().__init__()
            Cnn10 = eval(model_type_cnn10)
            cnn10 = Cnn10(sample_rate=sample_rate, window_size=window_size, 
                hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                classes_num=classes_num)

            checkpoint_cnn10 = torch.load(checkpoint_path_cnn10, map_location=device)
            cnn10.load_state_dict(checkpoint_cnn10['model'])
            if out_size != None:
                cnn10.fc_audioset.out_features = out_size
            cnn10.to(device)
            cnn10 = torch.nn.DataParallel(cnn10)
            self.model = cnn10
            self.loss_fn = torch.nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer


    epochs = 3
    nb_train = 1000

    dataset = CommonVoiceDataset(version="age")
    train, val = random_split(dataset, [nb_train, len(dataset) - nb_train])

    model = PL(max(dataset.df["age"]) + 1)
    trainer = pl.Trainer(max_epochs=3, gpus=1)
    trainer.fit(model,
                DataLoader(train, batch_size=64, shuffle=True, num_workers=56, pin_memory=True),
                DataLoader(val, batch_size=64, shuffle=True, num_workers=56, pin_memory=True))

    argmax = True
    dataloader = DataLoader(dataset, batch_size=64, num_workers=56, pin_memory=True)
    y_pred = []
    y_true = []
    model = model.eval().cuda()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.cuda()
            y = y.cuda()
            if argmax:
                out = np.argmax(model(x).cpu().numpy(), axis=1)
            else:
                out = model(x).cpu().numpy()[:, 1]
            y_pred.append(out)
            y_true.append(y.cpu().numpy())
    y_pred_age = np.concatenate(y_pred)
    y_true_age = np.concatenate(y_true)


    epochs = 3
    nb_train = 3000

    dataset = CommonVoiceDataset(version="gender")
    train, val = random_split(dataset, [nb_train, len(dataset) - nb_train])

    model = PL(max(dataset.df["gender"]) + 1)
    trainer = pl.Trainer(max_epochs=3, gpus=1)
    trainer.fit(model,
                DataLoader(train, batch_size=64, shuffle=True, num_workers=56, pin_memory=True),
                DataLoader(val, batch_size=64, shuffle=True, num_workers=56, pin_memory=True))

    argmax = False
    dataloader = DataLoader(dataset, batch_size=64, num_workers=56, pin_memory=True)
    y_pred = []
    y_true = []
    model = model.eval().cuda()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.cuda()
            y = y.cuda()
            if argmax:
                out = np.argmax(model(x).cpu().numpy(), axis=1)
            else:
                out = model(x).cpu().numpy()[:, 1]
            y_pred.append(out)
            y_true.append(y.cpu().numpy())
    y_pred_gender = np.concatenate(y_pred)
    y_true_gender = np.concatenate(y_true)


    query = AgeAggregateQuery(index)
    query.execute_metrics(err_tol=0.01, confidence=0.05, trials=1, y=[y_pred_age, y_true_age])

    query = GenderAggregateQuery(index)
    query.execute_metrics(err_tol=0.01, confidence=0.05, trials=1, y=[y_pred_gender, y_true_gender])

    query = AgeLimitQuery(index)
    query.execute_metrics(want_to_find=2, nb_to_find=10, GAP=0, y=[y_pred_age, y_true_age])

    query = AgePrecisionQuery(index)
    query.execute_metrics(10000, y=[y_pred_age, (y_true_age > 0).astype(int)])
    
    query = GenderPrecisionQuery(index)
    query.execute_metrics(10000, y=[y_pred_gender, y_true_gender])

    query = AgeRecallQuery(index)
    query.execute_metrics(10000, y=[y_pred_age, (y_true_age > 0).astype(int)])

    query = GenderRecallQuery(index)
    query.execute_metrics(10000, y=[y_pred_gender, y_true_gender])
