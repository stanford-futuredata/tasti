import numpy as np

class Query:
    def __init__(self, index):
        self.index = index
    
    def init(self):
        raise NotImplementedError
    
    def score(self, target_dnn_output):
        return len(target_dnn_output)
        
    def propagate(self, target_dnn_outputs, reps, topk_reps, topk_distances):
        y_true = np.zeros(len(topk_reps))
        y_pred = np.zeros(len(topk_reps))
        
        for i in range(len(target_dnn_outputs)):
            if target_dnn_outputs[i] != None:
                y_true[i] = self.score(target_dnn_outputs[i])

        for i in range(len(y_pred)):
            weights = topk_distances[i]
            weights = weights / weights.sum()
            y_pred[i] =  np.sum(y_true[topk_reps[i]] * weights)

        return y_pred
    
    def execute(self):
        y_pred = self.propagate(
            self.index.target_dnn_outputs,
            self.index.reps,
            self.index.topk_reps,
            self.index.topk_dists
        )
        print(y_pred.shape)
        print(y_pred)
        print(y_pred.sum())