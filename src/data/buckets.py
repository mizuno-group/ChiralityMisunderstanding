import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from .batch import make_batch_masks

class Buckets:
    def __init__(self, buckets, masks=None, seed=0):
        """
        A class containing multiple set of multiple features

        Parameters
        ----------
        buckets: list (n_buckets) of list (n_features) of np.array (n_data, *)
            Datasets.
        masks: array_like (n_buckets) of np.array (n_data)
            Masks that indicate which datapoint can be used in this multidatasets for each dataset.
            If None, all datas will be used.
        seed: int
            seed of random number generator used for splitting, shuffling, etc.
        """
        if masks is not None:
            assert len(buckets) == len(masks)

        self.buckets = []
        self.masks = []
        for i, bucket in enumerate(buckets):
            assert len(bucket) == len(buckets[0])
            for feature in bucket:
                assert len(feature) == len(bucket[0])
            
            if len(bucket[0]) > 0:
                self.buckets.append(bucket)
                if masks is None:
                    self.masks.append(np.arange(len(bucket[0])))
                else:
                    self.masks.append(masks[i])
        self.rstate = np.random.RandomState(seed=seed)
        self.dataset_masks = None
        self.shuffle_epoch = None
        
    def shuffle_batch(self, batch_size=None, n_tokens=None, n_epoch=1, i_length_feature=0):
        """
        Decides orders of iteration of batches in self.iterbatches()

        Parameters
        ----------
        batch_size: int or None
            Number of samples included in one batch.
            If None, batch_size is decided based on n_tokens.
        n_tokens: int or None
            Number of tokens included in one batch.
            Ignored when batch_size is not None.
        n_epoch: int
            Number of iterations of whole data to decide order.
        length_dim: int 
            Dimension of dataset of self which represents length of token.
        """
        show_log = len(self) >= 1000000
        if show_log:
            print("Shuffling batch...")

        dataset_masks = []
        for epoch in range(n_epoch):
            for i, mask in (tqdm(enumerate(self.masks)) if show_log else enumerate(self.masks)) :
                if batch_size is None:
                    batch_size_ex = n_tokens // self.buckets[i][i_length_feature].shape[1]
                else:
                    batch_size_ex = batch_size
                batch_masks = make_batch_masks(mask, batch_size=batch_size_ex,
                    random_state=self.rstate.randint(100), residue='include', shuffle=True)
                dataset_masks += [(i, batch_mask) for batch_mask in batch_masks]
        self.dataset_masks = np.array(dataset_masks, dtype=object)
        self.rstate.shuffle(self.dataset_masks)

    def iterbatches(self, n_shuffle_epoch=1, iterfeatures=None):
        if self.dataset_masks is None:
            raise ValueError("Please call shuffle_batch() before calling iterbatches().")
        if iterfeatures is None:
            iterfeatures = np.arange(len(self.buckets[0]))
        self.shuffle_epoch = 0
        while self.shuffle_epoch < n_shuffle_epoch:
            for i_bucket, mask in self.dataset_masks:
                yield tuple([self.buckets[i_bucket][i_feature][mask] for i_feature in iterfeatures])
            self.shuffle_epoch += 1

    def iterpoints(self, iterfeatures=None):
        if iterfeatures is None:
            iterfeatures = np.arange(len(self.buckets[0]))
        for bucket, mask in zip(self.buckets, self.masks):
            for i_point in mask:
                yield tuple([bucket[i_feature][i_point] for i_feature in iterfeatures])

    def split(self, ratios=None, split_array=False):
        """
        Split datasets into several multidatasets.

        Parameters
        ----------
        ratios: array_like
            Ratios of each splitted datasets.
        split_array: bool
            If True, array of dataset is splitted. If False, only masks are splitted.

        Returns
        -------
        multidatasets: tuple of MultiDataset
            Splitted MultiDatasets.
        """
        ratios = np.array(ratios, dtype=float)
        ratios = ratios / ratios.sum()
        n_split = len(ratios)
        self.split_masks = [[] for i in range(len(ratios))] #(n_splits, n_buckets)
        n_split_prev = np.zeros(n_split)
        n_split_total = np.zeros(n_split)
        n_total = 0
        for mask_whole in self.masks:
            n_split_prev = n_split_total.copy()
            n_total += len(mask_whole)
            n_split_total = np.floor(ratios*n_total)
            n_split_residue = ratios*n_total - n_split_total
            add_total = n_total - np.sum(n_split_total)
            add_split = np.argsort(-n_split_residue) < add_total
            n_split_total[add_split] += 1
            n_split_set = (n_split_total - n_split_prev).astype(np.int32)
            assert n_split_set.sum() == len(mask_whole)
            self.rstate.shuffle(mask_whole)
            split_begin = 0
            for i_split in range(n_split):
                self.split_masks[i_split].append(mask_whole[split_begin:split_begin+n_split_set[i_split]].copy())
                split_begin += n_split_set[i_split]
        buckets_splits = []
        if split_array:
            for masks in self.split_masks:
                buckets_splits.append(Buckets([[feature[mask] for feature in bucket]
                    for bucket, mask in zip(self.buckets, masks)], seed=self.rstate.randint(100)))
        else:
            for masks in self.split_masks:
                buckets_splits.append(Buckets(self.buckets, masks, seed=self.rstate.randint(100)))
        return tuple(buckets_splits)

    def __len__(self):
        return sum([len(mask) for mask in self.masks])

    def save(self, dirname):
        """
        Save bucket to a directory.

        Parameters
        ----------
        dirname: str
            Name of directory where datas are saved.
        """
        os.makedirs(dirname, exist_ok=True)
        for i_buc, bucket in enumerate(self.buckets):
            for i_feat, feature in enumerate(bucket):
                np.save(dirname+f"/bucket{i_buc}_feature{i_feat}.npy", feature)
        for i_buc, mask in enumerate(self.masks):
            np.save(dirname+f"/mask{i_buc}.npy", mask)
        with open(dirname+"/rstate.pkl", mode='wb') as f:
            pickle.dump(self.rstate, f)
        np.save(dirname+"/dataset_masks.npy", self.dataset_masks)
        params = pd.Series({"shuffle_epoch": self.shuffle_epoch,
         "n_bucket": len(self.buckets), 
         "n_feature": len(self.buckets[0])}, name='param')
        params.to_csv(dirname+"/params.csv")

    @classmethod
    def load(cls, dirname):
        """
        Load Buckets from datas saved by Buckets.save()

        Parameters
        ----------
        dirname: str
            Name of directory where datas are saved.
        
        """
        params = pd.read_csv(dirname+"/params.csv", index_col=0)['param']
        answer = Buckets([], seed=0)
        answer.buckets = []
        answer.masks = []
        for i_buc in range(int(params["n_bucket"])):
            bucket = [np.load(dirname+f"/bucket{i_buc}_feature{i_feat}.npy", allow_pickle=True)
                for i_feat in range(int(params['n_feature']))]
            answer.buckets.append(bucket)
            answer.masks.append(np.load(dirname+f"/mask{i_buc}.npy"))
        with open(dirname+"/rstate.pkl", mode='rb') as f:
            answer.rstate = pickle.load(f)
        answer.dataset_masks = np.load(dirname+"/dataset_masks.npy", allow_pickle=True)
        answer.shuffle_epoch = params["shuffle_epoch"]
        return answer
        
def MakeBuckets(bucketing_datas, pad_token,
    sub_bucketing_datas=[], sub_pad_tokens=[],
    datas=[], min_len=0, max_len=np.inf,
    min_bucket_step=1, min_n_token=1,
    residue='include', seed=0):
    """
    Parameters
    ----------
    bucketing_datas: list (n_features, n_data, length)
        Datas whose lengths are used for bucket classification (like random/canonical SMILES)
    sub_bucketing_datas: list (n_features, n_data, length)
        Datas whose length are not used for bucket classification but made into array.
    datas: list (n_features) of np.array (n_data, *)
        Additional datas (like biological features)
    min_len: int
        datapoints which len(point) < min_len will not be included into any bucket.
    max_len: int
        datapoints which len(point) > max_len will not be included into any bucket.
    min_bucket_step: int
        Width of each bucket must be longer than min_bucket_step
    pad_token: int
    sub_pad_tokens: list (len(sub_bucketing_datas)) of int
        Padding tokens for each of sub_bucketing_datas
    
    residue: str, 'ignore' or 'new'
        How to treat with residual largest datas.
        'ignore': don't include largest datas.
        'new': make new buckets from largest datas.
        'include': include datas to largest bucket.

    Returns
    -------
    bucket: MultiArray
        bucketed datasets. bucket.datasets[i].shape == (n_data, 2, max_len)
    """
    for data in list(bucketing_datas)+list(sub_bucketing_datas)+list(datas):
        assert len(data) == len(bucketing_datas[0])

    len_data = np.array([[len(data) for data in feature] for feature in bucketing_datas], dtype=np.int32)
    max_len_data = np.max(len_data, axis=0)
    min_len = int(max(min_len, np.min(max_len_data)))
    max_len = int(min(max_len, np.max(max_len_data)))

    len_index = [
        np.where(max_len_data == l)[0] for l in range(max_len+1)]
    len_index_len = [len(len_ind) for len_ind in len_index]
    
    bucket_ends = []
    bucket_start = min_len
    for l in range(min_len, max_len):
        if (sum(len_index_len[bucket_start:l+1])*l >= min_n_token and l-bucket_start+1 >= min_bucket_step):
            bucket_ends.append(l)
            bucket_start = l+1
    if residue == 'new':
        bucket_ends.append(max_len)
    elif residue == 'include':
        if len(bucket_ends) == 0:
            bucket_ends = [max_len]
        else:
            bucket_ends[-1] = max_len
    
    buckets = []
    bucket_start = min_len
    for bucket_end in bucket_ends:
        
        buc_mask = np.concatenate(len_index[bucket_start:bucket_end+1])
        n_data_buc = len(buc_mask)

        bucket = []
        for feature in bucketing_datas:
            feature_array = np.full((n_data_buc, bucket_end), fill_value=pad_token, dtype=type(feature[0][0]))
            for i_in_buc, i_data in enumerate(buc_mask):
                feature_array[i_in_buc][:len(feature[i_data])] = np.array(feature[i_data])
            bucket.append(feature_array)
        for sub_pad_token, feature in zip(sub_pad_tokens, sub_bucketing_datas):
            max_len = max([len(feature[i_data]) for i_data in buc_mask])
            feature_array = np.full((n_data_buc, max_len), fill_value=sub_pad_token, dtype=type(feature[0][0]))
            for i_in_buc, i_data in enumerate(buc_mask):
                feature_array[i_in_buc][:len(feature[i_data])] = np.array(feature[i_data])
            bucket.append(feature_array)
        for feature in datas:
            feature_array = np.ndarray([n_data_buc]+list(feature.shape[1:]), dtype=feature.dtype)
            for i_in_buc, i_data in enumerate(buc_mask):
                feature_array[i_in_buc] = feature[i_data]
            bucket.append(feature_array)
        bucket_start = bucket_end+1
        buckets.append(bucket)

    return Buckets(buckets, seed=seed)