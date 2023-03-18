import sys
sys.path.append('.')

import yaml
import glob
import pytest
import numpy as np
import time

import mindspore as ms
import mindocr
from mindocr.data import build_dataset
from mindocr.data.det_dataset import DetDataset
from mindocr.data.transforms.transforms_factory import transforms_dbnet_icdar15
from mindocr.data.rec_dataset import RecDataset
from mindspore import load_checkpoint, load_param_into_net

from mindocr.utils.visualize import show_img, draw_bboxes, show_imgs, recover_image

@pytest.mark.parametrize('task', ['det', 'rec'])
#@pytest.mark.parametrize('phase', ['train', 'eval'])
def test_build_dataset(task='det', phase='train', verbose=True, visualize=False):
    if task == 'rec':
        yaml_fp = 'configs/rec/test.yaml'
    else:
        yaml_fp = 'configs/det/test.yaml'

    with open(yaml_fp) as fp:
        cfg = yaml.safe_load(fp)

    if task == 'rec':
        from mindocr.data.transforms.rec_transforms import RecCTCLabelEncode
        dict_path = cfg['common']['character_dict_path']
        # read dict path and get class nums
        rec_info = RecCTCLabelEncode(character_dict_path=dict_path)
        #config['model']['head']['out_channels'] = num_classes
        print('=> num classes (valid chars + special tokens): ', rec_info.num_classes)

    dataset_config = cfg[phase]['dataset']
    loader_config = cfg[phase]['loader']


    dl = build_dataset(dataset_config, loader_config, is_train=(phase=='train'))
    dl.repeat(300)
    num_batches = dl.get_dataset_size()

    ms.set_context(mode=0)
    #batch = next(dl.create_tuple_iterator())
    num_tries = 100
    start = time.time()
    times = []
    iterator = dl.create_dict_iterator()
    for i, batch in enumerate(iterator):
        times.append(time.time()-start)
        if i >= num_tries:
            break

        if verbose:
            for k,v in batch.items():
                print(k, v.shape)
                if len(v.shape)<=2:
                    print(v)

        if (i == num_tries -1) and visualize:
            if task == 'det' and phase == 'eval':
                img = batch['image'][0].asnumpy()
                polys = batch['polys'][0].asnumpy()
                img_polys = draw_bboxes(recover_image(img), polys)
                show_img(img_polys)
        
        start = time.time()

    WU = 2
    tot = sum(times[WU:]) # skip warmup
    mean = tot / (num_tries-WU)
    print('Avg batch loading time: ', mean)

#@pytest.mark.parametrize('model_name', all_model_names)
def test_det_dataset():
    data_dir = '/data/ocr_datasets/ic15/text_localization/train'
    annot_file = '/data/ocr_datasets/ic15/text_localization/train/train_icdar15_label.txt'
    transform_pipeline = transforms_dbnet_icdar15(phase='train')
    ds = DetDataset(is_train=True, data_dir=data_dir, label_file=annot_file, sample_ratio=0.5, transform_pipeline=transform_pipeline, shuffle=False)

    print('num data: ', len(ds))
    for i in [223]:
        data_tuple = ds.__getitem__(i)

        # recover data from tuple to dict
        data = {k:data_tuple[i] for i, k in enumerate(ds.get_column_names())}

        print(data.keys())
        #print(data['image'])
        print(data['img_path'])
        print(data['image'].shape)
        print(data['polys'])
        print(data['texts'])
        #print(data['mask'])
        #print(data['threshold_map'])
        #print(data['threshold_mask'])
        for k in data:
            print(k, data[k])
            if isinstance(data[k], np.ndarray):
                print(data[k].shape)

        #show_img(data['image'], 'BGR')
        #result_img1 = draw_bboxes(data['ori_image'], data['polys'])
        img_polys = draw_bboxes(recover_image(data['image']), data['polys'])
        #show_img(result_img2, show=False, save_path='/data/det_trans.png')

        mask_polys= draw_bboxes(data['shrink_mask'], data['polys'])
        thrmap_polys= draw_bboxes(data['threshold_map'], data['polys'])
        thrmask_polys= draw_bboxes(data['threshold_mask'], data['polys'])
        show_imgs([img_polys, mask_polys, thrmap_polys, thrmask_polys], show=False, save_path='/data/ocr_ic15_debug2.png')


        np.savez('./det_db_label_samples.npz',
                image=data['image'],
                polys=data['polys'],
                texts=data['texts'],
                ignore_tags=data['ignore_tags'],
                shrink_map=data['shrink_map'],
                shrink_mask=data['shrink_mask'],
                threshold_map=data['threshold_map'],
                threshold_mask=data['threshold_mask'],
                )

        # TODO: check transformed image and label correctness

def test_rec_dataset(visualize=True):

    yaml_fp = 'configs/rec/crnn_icdar15.yaml'
    with open(yaml_fp) as fp:
        cfg = yaml.safe_load(fp)

    data_dir = '/Users/Samit/Data/datasets/ic15/rec/ch4_training_word_images_gt'
    label_path = '/Users/Samit/Data/datasets/ic15/rec/rec_gt_train.txt'
    ds = RecDataset(is_train=True,
            data_dir=data_dir,
            label_file=label_path,
            sample_ratio=1.0,
            shuffle = False,
            transform_pipeline = cfg['train']['dataset']['transform_pipeline'],
            output_keys = None)
    # visualize to check correctness
    from mindocr.utils.visualize import show_img, draw_bboxes, show_imgs, recover_image
    print('num data: ', len(ds))
    for i in [3]:
        data_tuple = ds.__getitem__(i)
        print('output columns: ', ds.get_column_names())
        # recover data from tuple to dict
        data = {k:data_tuple[i] for i, k in enumerate(ds.get_column_names())}

        print(data['img_path'])
        print(data['image'].shape)
        print('text: ', data['text'])
        print(f'\t Shapes: ', {k: data[k].shape for k in data if isinstance(data[k], np.ndarray)})
        print('label: ', data['label_ace'])
        print('label_ace: ', data['label_ace'])
        image = recover_image(data['image'])
        show_img(image, show=True) #, save_path='/data/ocr_ic15_debug2.png')

    '''
    dl = build_dataset(
            cfg['train']['dataset'],
            cfg['train']['loader'],
            is_train=True)
    batch = next(dl.create_dict_iterator())
    print(len(batch))
    for item in batch:
        print(item.shape)
    '''


if __name__ == '__main__':
    #test_build_dataset(task='det', phase='eval', visualize=True)
    #test_build_dataset(task='det', phase='train', visualize=False)
    test_build_dataset(task='rec', phase='train', visualize=False)
    #test_build_dataset(task='rec')
    #test_det_dataset()
    #test_rec_dataset()
