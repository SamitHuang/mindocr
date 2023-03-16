import os
from typing import Union, List, Tuple
from .base_dataset import BaseDataset


__all__ = ['DetDataset']


class DetDataset(BaseDataset):
    def __init__(self,
	    is_train: bool = True,
	    data_dir: Union[str, List[str]] = None, 
	    label_file: Union[List, str] = None,
	    sample_ratio: Union[List, float] = 1.0,
	    shuffle: bool = None,
	    **kwargs,):
	super().__init__(data_dir=data_dir, label_file=label_file)
	self.data_dir = data_dir
	assert isinstance(shuffle, bool), f'type error of {shuffle}'
        if isinstance(sample_ratio, float):
            sample_ratio = [sample_ratio] * len(label_file)
        shuffle = shuffle if shuffle is not None else is_train
	
	# load to self._data
        self.load_data_list(label_file, sample_ratio, shuffle)
	self.output_column_names = []


    def __getitem__(self, index):
	img_path, annot = self._data[index]

	return img_path, annot


    def get_column_names(self):
	return ['img_path', 'annot']
	

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
        for idx, annot_file in enumerate(label_file):
	    img_dir = data_dir[idx]
            with open(annot_file, "r", encoding='utf-8') as f:
                lines = f.readlines()
                if shuffle:
                    lines = random.sample(lines,
                                          round(len(lines) * sample_ratio[idx]))
                else:
                    lines = lines[:round(len(lines) * sample_ratio[idx])]
		
		for line in lines:	
            	    img_name, annot = self._parse_annotation(data_line)
        	    img_path = os.path.join(img_dir, img_name)
        	    assert os.path.exists(img_path), "{} does not exist!".format(img_path)

		    self._data.append((img_path, annot))


    def _parse_annotation(self, data_line: str):
        """
        Initially parse a data line string into a data dict containing input (img path) and label info (json dump).
        The label info will be encoded in transformation.
        """
        img_name, annot_str = data_line.strip().split('\t')

        return img_name, annot_str

