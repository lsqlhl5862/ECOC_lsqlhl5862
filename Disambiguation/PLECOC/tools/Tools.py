import numpy as np
import random


def tr_ts_split_idx(data, ts_size_percent=0.1,tv_size_percent=0.2):
    data_size = data.shape[0]
    ts_size = int(np.ceil(data_size * ts_size_percent))
    tv_size= int(np.ceil(data_size * tv_size_percent))
    tr_size = data_size - ts_size-tv_size
    data_idx = list(range(data_size))
    tr_idx = random.sample(data_idx, tr_size)
    rest_list=list(set(data_idx) - set(tr_idx))
    random.shuffle(rest_list)
    ts_idx = rest_list[:ts_size]
    tv_idx= rest_list[ts_size:]
    return tr_idx, ts_idx,tv_idx
