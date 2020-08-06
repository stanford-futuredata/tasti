import numpy as np
import tqdm
from numba import njit, prange

class Bucketter(object):
    def __init__(
            self,
            nb_buckets: int,
            seed: int=1
    ):
        self.nb_buckets = nb_buckets
        self.rand = np.random.RandomState(seed)

    def topk(self, k, dists):
        topks = []
        for i in tqdm.tqdm(range(len(dists))):
            topks.append(np.argpartition(dists[i], k)[0:k])
        return np.array(topks)

@njit(parallel=True)
def get_dists(x, embeddings):
    dists = np.zeros(len(embeddings), dtype=np.float32)
    for i in prange(len(embeddings)):
        dists[i] = np.sqrt(np.sum((x - embeddings[i]) ** 2))
    return dists

@njit(parallel=True)
def get_and_update_dists(x, embeddings, min_dists):
    dists = np.zeros(len(embeddings), dtype=np.float32)
    for i in prange(len(embeddings)):
        dists[i] = np.sqrt(np.sum((x - embeddings[i]) ** 2))
        if dists[i] < min_dists[i]:
            min_dists[i] = dists[i]
    return dists

class FPFBucketter(Bucketter):
    def bucket(
            self,
            embeddings: np.ndarray,
            max_k: int
    ):
        reps = np.full(self.nb_buckets, -1)
        reps[0] = self.rand.randint(len(embeddings))
        min_dists = np.full(len(embeddings), np.Inf, dtype=np.float32)
        dists = np.zeros((self.nb_buckets, len(embeddings)))
        dists[0, :] = get_and_update_dists(
                embeddings[reps[0]],
                embeddings,
                min_dists
        )

        for i in tqdm.tqdm(range(1, self.nb_buckets), desc='FPFBucketter'):
            reps[i] = np.argmax(min_dists)
            dists[i, :] = get_and_update_dists(
                    embeddings[reps[i]],
                    embeddings,
                    min_dists
            )
            
        dists = dists.transpose()
        
        topk_reps = self.topk(max_k, dists)
        topk_dists = np.zeros([len(topk_reps), max_k])
        
        for i in range(len(topk_dists)):
            topk_dists[i] = dists[i, topk_reps[i]]
            
        return reps, topk_reps, topk_dists