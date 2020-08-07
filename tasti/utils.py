import tasti
import numpy as np
import torch, torchvision

"""
Both BlazeIt and SUPG assume for the sake of fast experiments that you have access to all of the Target DNN outputs.
These classes will allow you to still use the BlazeIt and SUPG algorithms by executing the Target DNN in realtime.
"""

class DNNOutputCache:
    def __init__(self, target_dnn, dataset, target_dnn_callback=lambda x: x):
        target_dnn.cuda()
        target_dnn.eval()
        self.target_dnn = target_dnn
        self.dataset = dataset
        self.target_dnn_callback = target_dnn_callback
        self.length = len(dataset)
        self.cache = [None]*self.length
        self.nb_of_invocations = 0
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.cache[idx] == None:
            with torch.no_grad():
                record = self.dataset[idx].unsqueeze(0).cuda()
                result = self.target_dnn(record)
            result = self.target_dnn_callback(result)
            self.cache[idx] = result
            self.nb_of_invocations += 1
        return self.cache[idx]
            
class DNNOutputCacheFloat:
    def __init__(self, target_dnn_cache, scoring_fn, idx):
        self.target_dnn_cache = target_dnn_cache
        self.scoring_fn = scoring_fn
        self.idx = idx
        
        def override_arithmetic_operator(name):
            def func(self, *args):
                value = self.target_dnn_cache[self.idx]
                value = self.scoring_fn(value)
                value = np.float32(value)
                args_f = []
                for arg in args:
                    if type(arg) is tasti.utils.DNNOutputCacheFloat:
                        arg = np.float32(arg)
                    args_f.append(arg)
                value = getattr(value, name)(*args_f)
                return value 
            return func
        
        operator_names = [
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__", 
            "__neg__", 
            "__pos__", 
            "__radd__",
            "__rmul__",
        ]
            
        for name in operator_names:
            setattr(DNNOutputCacheFloat, name, override_arithmetic_operator(name))
        
    def __repr__(self):
        return f'DNNOutputCacheFloat(idx={self.idx})'
    
    def __float__(self):
        value = self.target_dnn_cache[self.idx]
        value = self.scoring_fn(value)
        return float(value)