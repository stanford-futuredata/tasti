# Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data

This is the official project page for "Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data"

Please read the [paper](https://google.com) for full technical details.

# Requirements

For requirements, take a look at requirements.txt. You can automatically install everything with:
`pip install -r requirements.txt`

To reproduce the experiments regarding the video `night-street` your machine will need:
- 300+GB of memory
- 500+GB of space
- A GPU (this has only been tested with NVIDIA P100 and V100)

Hardware requirements will vary depending on the dataset and hyperparameters.

# Quickstart

You can download the `night-street` video data [here](https://google.com). You will need to install [swag](https://google.com) to use it appropriately. If you don't want to download the entire video, we also provide pre-computed embeddings [here](https://google.com).

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




