import os
from typing import Union, List, Tuple
import random
import numpy as np
from .base_dataset import BaseDataset


__all__ = ['DetDataset']


class DetDataset(BaseDataset):
    def __init__(self,
	    is_train: bool = True,
	    data_dir: Union[str, List[str]] = None,
	    label_file: Union[str, List[str]] = None,
	    sample_ratio: Union[List, float] = 1.0,
	    shuffle: bool = None,
	    **kwargs,):
        super().__init__(data_dir=data_dir, label_file=label_file)

        assert isinstance(shuffle, bool), f'type error of {shuffle}'
        if isinstance(sample_ratio, float):
            sample_ratio = [sample_ratio] * len(label_file)
        shuffle = shuffle if shuffle is not None else is_train

	    # load to self._data
        self.load_data_list(self.label_file, sample_ratio, shuffle)

        self.output_columns = ['img_path', 'label']
        #self.output_columns = ['img_bytes', 'label']

    def __getitem__(self, index):
        img_path, label= self._data[index]

        return img_path, label
        #img_bytes = self._load_image_bytes(img_path)
        #return img_bytes, label


    def load_data_list(self, label_file: List[str], sample_ratio: List[float], shuffle: bool = False,
                       **kwargs) -> List[dict]:
        """ Load data list from label_file which contains infomation of image paths and annotations
        Args:
            label_file: annotation file path(s)
            sample_ratio sample ratio for data items in each annotation file
            shuffle: shuffle the data list

        Returns:
            data (List[dict]): A list of annotation dict, which contains keys: img_path, annot...
        """

        # parse image file path and annotation and load into self._data
        for idx, label_fp in enumerate(label_file):
            img_dir = self.data_dir[idx]
            with open(label_fp, "r", encoding='utf-8') as f:
                lines = f.readlines()
                if shuffle:
                    lines = random.sample(lines,
                                          round(len(lines) * sample_ratio[idx]))
                else:
                    lines = lines[:round(len(lines) * sample_ratio[idx])]

                for line in lines:
                    img_name, annot_str = self._parse_annotation(line)
                    img_path = os.path.join(img_dir, img_name)
                    assert os.path.exists(img_path), "{} does not exist!".format(img_path)
                    self._data.append((img_path, annot_str))


    def _parse_annotation(self, data_line: str):
        """
        Initially parse a data line string into a data dict containing input (img path) and label info (json dump).
        The label info will be encoded in transformation.
        """
        img_name, annot_str = data_line.strip().split('\t')

        return img_name, annot_str

if __name__ == '__main__':
    detds = DetDataset(True, '/Users/Samit/Data/datasets/ic15/det/train/ch4_training_images',
                '/Users/Samit/Data/datasets/ic15/det/train/det_gt.txt',
                shuffle=False
               )
    #print(next(detds))
    import mindspore as ms
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    ds = ms.dataset.GeneratorDataset(
                    source=detds,
                    column_names=['img_bytes', 'label']
                    )

    class SimpleDecoder():
        def __init__(self):
            pass
        def __call__(self, *data):
            raw_img = data[0]
            label = data[1]
            #print(data[0].dtype.type, data[1].dtype.type)
            img = np.frombuffer(raw_img, dtype='uint8') # bytes of raw img file data
            img = cv2.imdecode(img, cv2.IMREAD_COLOR) # decoded (or uncompressed) image data
            return img, label

    ds = ds.map(operations=[SimpleDecoder()], input_columns=['img_bytes', 'label'])
    img, label = next(ds.create_tuple_iterator())
    #print(next(ds.create_tuple_iterator()))

    img = img.asnumpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
    plt.show()

