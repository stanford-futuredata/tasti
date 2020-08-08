# Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data

This is the official project page for "Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data"

Please read the [paper](https://google.com) for full technical details.

# Requirements

For requirements, take a look at requirements.txt. You can install everything with:
`pip install -r requirements.txt`. You will also need (install via `pip install -e .`:
- [SWAG](https://github.com/stanford-futuredata/swag-python)
- [BlazeIt](https://github.com/stanford-futuredata/blazeit)
- [SUPG](https://github.com/stanford-futuredata/supg)

To reproduce the experiments regarding the video `night-street` your machine will need:
- 300+GB of memory
- 500+GB of space
- A GPU (this has only been tested with NVIDIA P100 and V100)

Hardware requirements will vary depending on the dataset and hyperparameters.

# Quickstart

We provide example code for creating a TASTI for the `night-street` video dataset along with several queries mentioned in the paper (Aggregation, Limit, SUPG). You can find the data available [here](https://drive.google.com/drive/u/1/folders/1rO2dJkHurbrKHf5cHtFra01uk5hlhHdO). If you are running the offline example (target dnn outputs for every frame in the video are precomputed), you will need to also download [this](https://drive.google.com/drive/u/1/folders/1rO2dJkHurbrKHf5cHtFra01uk5hlhHdO). For more details, read the annotated code in `tasti/examples/night_street_ofline.py`.

```
import tasti
config = tasti.examples.NightStreetOfflineConfig()
index = tasti.examples.NightStreetOfflineIndex(config)
index.init()

query = tasti.examples.NightStreetAggregateQuery(index)
result = query.execute()
print(result)

query = tasti.examples.NigthStreetSUPGPrecisionQuery(index)
result = query.execute()
print(result)

query = tasti.examples.NightStreetSUPGRecallQuery(index)
result = query.execute()
print(result)
```

We also provide an online version of the code that allows you to run the Target DNN in realtime. For efficieny purposes, we implement [Mask R-CNN ResNet-50 FPN](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection). However, the actual model used in the experiments of the paper is the Mask R-CNN X 152 model available in [detectron2](https://github.com/facebookresearch/detectron2).

# Customizing TASTI

Our code allows for you to create your own TASTI. You will have to sub-class the tasti.Index class and implement a few functions:

```
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
