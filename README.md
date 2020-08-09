# Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data

This is the official project page for "Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data"

Please read the [paper](https://google.com) for full technical details.

# Requirements

Install the requitements with `pip install -r requirements.txt`. You will also need (via `pip install -e .`):
- [SWAG](https://github.com/stanford-futuredata/swag-python)
- [BlazeIt](https://github.com/stanford-futuredata/blazeit)
- [SUPG](https://github.com/stanford-futuredata/supg)

To reproduce the experiments regarding the video `night-street` your machine will need:
- 300+GB of memory
- 500+GB of space
- GPU (e.g., NVIDIA V100, TITAN V)

Hardware requirements will vary depending on the dataset and hyperparameters.

# Reproducing Experiments

We provide code for creating a TASTI for the `night-street` video dataset along with several queries mentioned in the paper (aggregation, limit, SUPG, etc). You can download the `night-street` video dataset [here](https://drive.google.com/drive/folders/1phQuGu4oWwbArurprqruMztTdP1Fzz2F?usp=sharing). Download the `2017-12-14.zip` and `2017-12-17.zip` files. Unzip the files and place the video data in `/lfs/1/jtguibas/data` (feel free to change this path in night_street_offline.py). For speed purposes, the target dnn will not run in realtime and we have instead provided the outputs [here](https://drive.google.com/drive/folders/1phQuGu4oWwbArurprqruMztTdP1Fzz2F?usp=sharing). Place the csv files in `/lfs/1/jtguibas/data`. Then, you can reproduce the experiments by running:

```python
# night_street_offline.py

import tasti
config = tasti.examples.NightStreetOfflineConfig()
index = tasti.examples.NightStreetOfflineIndex(config)
index.init()

query = tasti.examples.NightStreetAggregateQuery(index)
query.execute_metrics()

query = tasti.examples.NightStreetLimitQuery(index)
query.execute_metrics(5)

query = tasti.examples.NightStreetSUPGPrecisionQuery(index)
query.execute_metrics()

query = tasti.examples.NightStreetSUPGRecallQuery(index)
query.execute_metrics()

query = tasti.examples.NightStreetLHSPrecisionQuery(index)
query.execute_metrics()

query = tasti.examples.NightStreetLHSRecallQuery(index)
query.execute_metrics()

query = tasti.examples.NightStreetAveragePositionAggregateQuery(index)
query.execute_metrics()
```

We also provide an online version of the code that allows you to run the target dnn in realtime. For efficieny purposes, we implement [Mask R-CNN ResNet-50 FPN](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection) with significantly less intensive hyperparameters. However, the actual model used in the experiments of the paper is the Mask R-CNN X 152 model available in [detectron2](https://github.com/facebookresearch/detectron2). We encourage you to replace the inference with TensorRT or another optimized model serving system for more serious needs.

To run the WikiSQL example, download the data [here](https://github.com/salesforce/WikiSQL). 

# Customizing TASTI

Our code allows for you to create your own TASTI. You will have to sub-class the `tasti.Index` class and implement a few functions:

```python
import tasti

class MyIndex(tasti.Index):
    def is_close(self, a, b):
        '''
        return a Boolean of whether records a and b are 'close'
        '''
        raise NotImplementedError
        
    def get_target_dnn_dataset(self):
        '''
        return a torch.utils.data.Dataset object
        '''
        raise NotImplementedError
    
    def get_embedding_dnn_dataset(self): 
        '''
        return a torch.utils.data.Dataset object
        '''
        raise NotImplementedError
        
    def get_target_dnn(self):
        '''
        return a torch.nn.Module object
        '''
        raise NotImplementedError
        
    def get_embedding_dnn(self):
        '''
        return a torch.nn.Module object
        '''
        raise NotImplementedError
        
    def target_dnn_callback(self, target_dnn_output):
        raise NotImplementedError
        
    def override_target_dnn_cache(self, target_dnn_cache):
        '''
        Optional:
        Allows for you to override the target_dnn_cache when you have the
        target dnn outputs already cached
        '''
        raise NotImplementedError
        
class MyQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
        
config = tasti.IndexConfig()
config.nb_buckets = 500

index = MyIndex(config)
index.init()

query = MyQuery()
result = query.execute()
print(result)
```

# Config
These are options available in `tasti.IndexConfig` which get passed into the `tasti.Index` object.
- `do_mining`, Boolean that determines whether the mining step is skipped or not
- `do_training`, Boolean that determines whether the training/fine-tuning step of the embedding dnn is skipped or not
- `do_infer`, Boolean that allows you to either compute embeddings on the spot or load them from `cache/embeddings.npy`
- `do_bucketting`, Boolean that allows you to compute the buckets or load them from `cache`.
- `batch_size`, general batch size for both the target and embedding dnn
- `train_margin`, controls the margin parameter of the triplet loss
- `max_k`, controls the k parameter described in the paper (for computing distance weighted means and votes)
- `nb_train`, controls how many datapoints are labled to perform the triplet training
- `nb_buckets`, controls the number of buckets used to construct the index
- `nb_training_its`, controls the number of datapoints are passed through the model during training
