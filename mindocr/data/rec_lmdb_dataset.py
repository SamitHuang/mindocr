from typing import Union, List
import numpy as np
import random
import lmdb
import os

from .transforms.transforms_factory import create_transforms, run_transforms
from .base_dataset import BaseDataset

class LMDBDataset(BaseDataset):
    """
    LMDB dataset.

    Args:
        is_train:
        data_dir:
        shuffle, Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
            -       if None, default transform pipeline for text detection will be taken.
        output_keys (list): required, indicates the keys in data dict that are expected to output for dataloader. if None, all data keys will be used for return.
        global_config: additional info, used in data transformation, possible keys:
            - character_dict_path

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a tuple for the specified data item.
        You can specify the `output_keys` arg to order the output data for dataloader.

    Notes:
        1. Dataset file structure should follow:
            data_dir
            ├── dataset01
                ├── data.mdb
                ├── lock.mdb
            ├── dataset02
                ├── data.mdb
                ├── lock.mdb
            ├── ...
    """
    def __init__(self,
            is_train: bool = True,
            data_dir: Union[str, List[str]] = '',
            sample_ratios: Union[float, List[float]] = 1.0,
            shuffle: bool = None,
            transform_pipeline: List[dict] = None,
            output_keys: List[str] = None,
            **kwargs,
            ):
        super().__init__(data_dir=data_dir)

        assert isinstance(shuffle, bool), f'type error of {shuffle}'
        shuffle = shuffle if shuffle is not None else is_train

        sample_ratio = sample_ratios[0] if isinstance(sample_ratios, list) else sample_ratios # TODO: support sample_ratio for each data dir
        self.lmdb_sets = self.load_list_of_hierarchical_lmdb_dataset(data_dir)
        self._data = self.dataset_traversal(sample_ratio, shuffle)

        self.output_columns = ['img_bytes', 'label']

    def load_list_of_hierarchical_lmdb_dataset(self, data_dir):
        if isinstance(data_dir, str):
            results = self.load_hierarchical_lmdb_dataset(data_dir)
        elif isinstance(data_dir, list):
            results = {}
            for sub_data_dir in data_dir:
                start_idx = len(results)
                lmdb_sets = self.load_hierarchical_lmdb_dataset(sub_data_dir, start_idx)
                results.update(lmdb_sets)
        else:
            results = {}

        return results

    def load_hierarchical_lmdb_dataset(self, data_dir, start_idx=0):

        lmdb_sets = {}
        dataset_idx = start_idx
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath":dirpath, "env":env, \
                    "txn":txn, "num_samples":num_samples}
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self, sample_ratio, shuffle):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num

        if shuffle:
            np.random.shuffle(data_idx_order_list)

        data_idx_order_list = data_idx_order_list[:round(len(data_idx_order_list) * sample_ratio)]

        return data_idx_order_list

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            label = None
        else:
            label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self._data[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        img_bytes, label = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'],
                                                file_idx)
        #if sample_info is None:
        #    return self.__getitem__(np.random.randint(self.__len__())) #TODO: why random?

        return img_bytes, label

    def __len__(self):
        return self._data.shape[0]
