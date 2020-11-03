import numpy as np
import random
import os
import time
from contextlib import contextmanager
import torch
import logging
import cloudpickle


def load_from_pkl(load_path):
    frb = open(load_path, 'rb')
    obj = cloudpickle.loads(frb.read())
    return obj


def save_as_pkl(obj, save_path):
    fwb = open(save_path, 'wb')
    fwb.write(cloudpickle.dumps(obj))
    return


def seed_everything(seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for i, col in enumerate(df.columns):
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
        except ValueError:
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


class Timer:
    def __init__(self):
        self.processing_time = 0

    @contextmanager
    def timer(self, name):
        logging.info(f'[{name}] start')
        t0 = time.time()
        yield
        t1 = time.time()
        processing_time = t1 - t0
        self.processing_time += round(processing_time, 2)
        if self.processing_time < 60:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time:.2f} sec)')
        elif self.processing_time < 3600:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 60:.2f} min)')
        else:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 3600:.2f} hour)')

    def get_processing_time(self):
        return round(self.processing_time, 2)