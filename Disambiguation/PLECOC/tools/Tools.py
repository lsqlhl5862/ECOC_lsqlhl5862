import numpy as np
import random


def tr_ts_split_idx(data, ts_size_percent=0.1):
    data_size = data.shape[0]
    ts_size = int(np.ceil(data_size * ts_size_percent))
    tr_size = data_size - ts_size
    data_idx = list(range(data_size))
    tr_idx = random.sample(data_idx, tr_size)
    ts_idx = list(set(data_idx) - set(tr_idx))
    return tr_idx, ts_idx
