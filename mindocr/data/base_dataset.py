from typing import Union, List, Tuple
import os


__all__ = ['BaseDataset']

class BaseDataset(object):
    '''
    Source dataset to parse data (and label) file.

    Note: 
        - Don't do heavy computation here such as image processing and augmentation. Leave it to mapping.
        - Support multiple datasets
    '''
    def __init__(self, 
                data_dir: Union[str, List[str]], 
                label_file: Union[str, List[str]] = None,
                **kwargs,
                ):

        self._index = 0
        self._data = [] 
        
        # check
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        for f in data_dir:
            if not os.path.exists(f):
                raise ValueError(f"{f} not existed. Please check the yaml file for both train and eval")
        self.data_dir = data_dir

        if label_file is not None:
            if isinstance(label_file, str):
                label_file = [label_file]
            for f in label_file:
                if not os.path.exists(f):
                    raise ValueError(f"{f} not existed. Please check the yaml file for both train and eval")
        self.label_file = label_file


    def __getitem__(self, index):
        #return self._data[index]
        raise NotImplementedError


    def get_output_columns(self) -> List[str]:
        '''
        get the column names for the output tuple of __getitem__, required for data mapping in the next step 
        '''
        raise NotImplementedError


    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = self.__getitem__(self._index)
            self._index += 1
            return item 


    def __len__(self):
        
        return len(self._data)


    #def _load_image(self, img_path):
    #    img = cv2.cvtColor(cv2.imread(img_path),
    #                       cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
    #   return np.ascontiguousarray(img).astype(np.float32)
