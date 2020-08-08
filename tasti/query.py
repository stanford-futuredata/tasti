import tasti
import numpy as np
import supg.datasource as datasource
from tqdm.autonotebook import tqdm
from blazeit.aggregation.samplers import ControlCovariateSampler
from supg.sampler import ImportanceSampler
from supg.selector import ApproxQuery
from supg.selector import RecallSelector, ImportancePrecisionTwoStageSelector

class BaseQuery:
    def __init__(self, index):
        self.index = index
        
    def score(self, target_dnn_output):
        raise NotImplementedError
        
    def propagate(self, target_dnn_cache, reps, topk_reps, topk_distances):
        score_fn = self.score
        y_true = np.array(
            [tasti.DNNOutputCacheFloat(target_dnn_cache, score_fn, idx) for idx in range(len(topk_reps))]
        )
        y_pred = np.zeros(len(topk_reps))

        for i in tqdm(range(len(y_pred)), 'Propagation'):
            weights = topk_distances[i]
            weights = weights / weights.sum()
            counts = y_true[topk_reps[i]]
            y_pred[i] =  np.sum(counts * weights)
        return y_pred, y_true
        
    def execute(self):
        raise NotImplementedError
           
class AggregateQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError
    
    def execute(self):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )
        
        err_tol = 0.01
        confidence = 0.05
        r = np.amax(np.rint(y_pred)) + 1
        sampler = ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r)
        estimate, nb_samples = sampler.sample()
        
        print('Results')
        print('=======')
        print('Initial Estimate:', y_pred.sum())
        print('Debiased Estimate:', estimate)
        print('Samples:', nb_samples)
        
        return {'initial_estimate': y_pred.sum(), 'debiased_estimate': estimate, 'samples': nb_samples}
    
class LimitQuery(BaseQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
    def execute(self, want_to_find, GAP=300):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )
        order = np.argsort(y_pred)[::-1]
        ret_inds = []
        visited = set()
        nb_calls = 0
        for ind in order:
            if ind in visited:
                continue
            nb_calls += 1
            if float(y_true[ind]) >= want_to_find:
                ret_inds.append(ind)
                for offset in range(-GAP, GAP+1):
                    visited.add(offset + ind)
            if len(ret_inds) >= 10:
                break
                
        print('Results')
        print('=======')
        print('Number of Calls:', nb_calls)
        print('Indexes:', ret_inds)
        return {'nb_calls': nb_calls, 'ret_inds': ret_inds}
    
class SUPGPrecisionQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError
    
    def execute(self):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )
        
        source = datasource.RealtimeDataSource(y_pred, y_true)
        sampler = ImportanceSampler()
        query = ApproxQuery(
            qtype='rt',
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=10000
        )
        selector = ImportancePrecisionTwoStageSelector(query, source, sampler)
        inds = selector.select()
        
        print('Results')
        print('=======')
        print('Initial Estimate:', y_pred.sum())
        print('Debiased Estimate:', inds.shape[0])
        print('idxs:', inds)
        print('shape:', inds.shape)
        
        return {'initial_estimate': y_pred.sum(), 'debiased_estimate': inds.shape[0], 'idxs': inds, 'shape': inds.shape}
    
class SUPGRecallQuery(SUPGPrecisionQuery):
    def execute(self):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )
        
        source = datasource.RealtimeDataSource(y_pred, y_true)
        sampler = ImportanceSampler()
        query = ApproxQuery(
            qtype='rt',
            min_recall=0.90, min_precision=0.90, delta=0.05,
            budget=10000
        )
        selector = RecallSelector(query, source, sampler, sample_mode='sqrt')
        inds = selector.select()
        
        print('Results')
        print('=======')
        print('Initial Estimate:', y_pred.sum())
        print('Debiased Estimate:', inds.shape[0])
        print('idxs:', inds)
        print('shape:', inds.shape)
        
        return {'initial_estimate': y_pred.sum(), 'debiased_estimate': inds.shape[0], 'idxs': inds, 'shape': inds.shape}