from typing import List
import os
import mindspore as ms
from addict import Dict
from .det_dataset import DetDataset
from .rec_dataset import RecDataset
from .rec_lmdb_dataset import LMDBDataset
from .transforms.transforms_factory import create_transforms
from .transforms.general_transforms import Compose 

supported_dataset_types = ['BaseDataset', 'DetDataset', 'RecDataset', 'LMDBDataset']

def build_dataset(
        dataset_config: dict,
        loader_config: dict,
        num_shards=None,
        shard_id=None,
        is_train=True,
        **kwargs,
        ):
    '''
    Build dataset

    Args:
        dataset_config (dict): dataset reading and processing configuartion containing keys:
            - type: dataset type, 'DetDataset', 'RecDataset'
            - dataset_root (str): the root directory to store the (multiple) dataset(s)
            - data_dir (Union[str, List[str]]): directory to the data, which is a subfolder path related to `dataset_root`. For multiple datasets, it is a list of subfolder paths.
            - label_file (Union[str, List[str]]): file path to the annotation related to the `dataset_root`. For multiple datasets, it is a list of relative file paths.
            - transform_pipeline (list[dict]): each element corresponds to a transform operation on image and/or label

        loader_config (dict): dataloader configuration containing keys:
            - batch_size: batch size for data loader
            - drop_remainder: whether to drop the data in the last batch when the total of data can not be divided by the batch_size
        num_shards: num of devices for distributed mode
        shard_id: device id
        is_train: whether it is in training stage

    Return:
        data_loader (Dataset): dataloader to generate data batch

    Notes:
        - The main data process pipeline in MindSpore contains 3 parts: 1) load data files and generate source dataset, 2) perform per-data-row mapping such as image augmentation, 3) generate batch and apply batch mapping. 
        - Each of the three steps supports multiprocess. Detailed machenism can be seen in https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore.dataset.html
        - A data row is a data tuple item containing multiple elements such as (image_i, mask_i, label_i). A data column corresponds to an element in the tuple like 'image', 'label'. 
        - The total number of `num_parallel_workers` used for data loading and processing should not be larger than the maximum threads of the CPU. Otherwise, it will lead to resource competing overhead. Especially for distributed training, `num_parallel_workers` should not be too large to avoid thread competition. 
    '''

    # check dataset_root, data_dir, and label_file and merge if dataset_root is given
    dataset_config = _validate_data_paths(dataset_config)

    # set default params for process pipeline
    #   Number of subprocesses used to fetch the dataset/map data row/gen batch in parallel
    num_workers = loader_config.get("num_workers", 8) 
    #   the length of the cache queue in the data pipeline for each worker, used to reduce waiting time. Larger value leads to more memory consumption. Default: 16 
    prefetch_size = loader_config.get("prefetch_size", 16) 
    ms.dataset.config.set_prefetch_size(prefetch_size)  
    #print('Prefetch size: ', ms.dataset.config.get_prefetch_size())
    #   MB of shared memory between processes to copy data
    max_rowsize =  loader_config.get("max_rowsize", 64) 

    #   TODO: find optimal setting automatically according to num of CPU cores
    # auto tune num_workers, prefetch. (This conflicts the profiler)
    #ms.dataset.config.set_autotune_interval(5)
    #ms.dataset.config.set_enable_autotune(True, "./dataproc_autotune_out")  


    # 1. create source dataset (GeneratorDataset)
    dataset_class_name = dataset_config.pop('type')
    assert dataset_class_name in supported_dataset_types, "Invalid dataset name"
    dataset_class = eval(dataset_class_name)

    dataset_args = dict(is_train=is_train, **dataset_config)
    dataset = dataset_class(**dataset_args)

    dataset_column_names = dataset.get_column_names()
    print('==> Dataset columns: \n\t', dataset_column_names)

    # generate source dataset (source w.r.t. the dataset.map pipeline) based on python callable numpy dataset in parallel 
    ds = ms.dataset.GeneratorDataset(
                    source=dataset,
                    column_names=dataset_column_names,
                    num_parallel_workers=max(1, num_workers//2), #2
                    num_shards=num_shards,
                    shard_id=shard_id,
                    #python_multiprocessing=Flase,
                    #max_rowsize =max_rowsize,
                    shuffle=loader_config['shuffle'],
                    )

    # 2. create transformation
    # data mapping, processing and augmentation (high-performance transformation)
    output_columns = dataset_config['output_keys']
    if dataset_config['transform_pipeline'] is not None:
        transform_list = create_transforms(dataset_config['transform_pipeline'])
        # compose and map to required input/output tuple format 
        operation = Compose(transform_list, input_columns=dataset_column_names, output_columns=output_columns)  
        ds = ds.map(operations=[operation],
                    input_columns=dataset_column_names,
                    output_columns=output_columns,
                    python_multiprocessing=True, # keep True to improve performace for heavy computation.
                    num_parallel_workers=num_workers,
                    max_rowsize =max_rowsize,
                    )
    else:
        opeartion = None

    # 3. create loader
    # get batch of dataset by collecting batch_size consecutive data rows and apply batch operations 
    drop_remainder = loader_config.get('drop_remainder', is_train)
    if is_train and drop_remainder == False:
        print('WARNING: drop_remainder should be True for training, otherwise the last batch may lead to training fail.')

    dataloader = ds.batch(
                    loader_config['batch_size'],
                    drop_remainder=drop_remainder,
                    num_parallel_workers=num_workers, # set depends on computation cost 
                    #per_batch_map=operation,
                    #python_multiprocessing=True, # set True for heavy computation
                    #max_rowsize =max_rowsize,
                    #input_columns=input_columns,
                    #output_columns=batch_column,
                    )

    #steps_pre_epoch = dataset.get_dataset_size()
    return dataloader


def _validate_data_paths(dataset_config):
    if 'dataset_root' in dataset_config:
        if isinstance(dataset_config['data_dir'], str):
            dataset_config['data_dir'] = os.path.join(dataset_config['dataset_root'], dataset_config['data_dir']) # to absolute path
        else:
            dataset_config['data_dir'] = [os.path.join(dataset_config['dataset_root'], dd) for dd in dataset_config['data_dir']]

        if 'label_file' in dataset_config:
            if isinstance(dataset_config['label_file'], str):
                dataset_config['label_file'] = os.path.join(dataset_config['dataset_root'], dataset_config['label_file'])
            else:
                dataset_config['label_file'] = [os.path.join(dataset_config['dataset_root'], lf) for lf in dataset_config['label_file']]

    return dataset_config
