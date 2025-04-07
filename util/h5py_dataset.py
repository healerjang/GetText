import h5py
import os
import numpy as np
from tqdm import tqdm
import atexit

def get_dtype(item):
    if isinstance(item, np.ndarray):
        dtype = item.dtype
    elif isinstance(item, (str, bytes)):
        dtype = h5py.string_dtype(encoding='utf-8')
    else:
        raise TypeError("Item type must be 'str' or 'np.ndarray'.")

    return dtype

class H5pyDatasetError(Exception):
    pass

class H5pyDatafile:
    def __init__(self, save_path, batch_size=512):
        self.save_path = save_path
        self.new_file = False if os.path.exists(save_path) else True
        self.batch_size = batch_size
        self.temporary_dic_write = {}
        self.temporary_dic_read = {}

        dir_path, filename = os.path.split(save_path)
        os.makedirs(dir_path, exist_ok=True)

        self.__create_class__()
        atexit.register(self.__exit__)

    def __exit__(self):
        print('class exit')
        self.__save_all__()

    def __create_class__(self):
        if not self.new_file:
            with h5py.File(self.save_path, 'r') as f:
                key_names = f['keyname'][...].tolist()
                key_names = [item.decode('utf-8') for item in key_names]
                self.temporary_dic_write['keyname'] = key_names

                for key in key_names:
                    dataset = f[key]
                    self.temporary_dic_write[key] = {
                        'data':[],
                        'index':0,
                        'dtype':get_dtype(dataset[0]),
                        'shape':dataset.shape,
                    }
                    self.temporary_dic_read[key] = {
                        'index':0,
                        'data':dataset[:self.batch_size],
                    }
        else:
            self.temporary_dic_write['keyname'] = []

    def __create_dataset__(self, key, item, shape, dt):
        with h5py.File(self.save_path, 'a') as f:
            dset = f.create_dataset(key, shape=shape, maxshape=(None, *shape[1:]), dtype=dt)
            dset[...] = item

    def __append_dataset__(self, key, item, append_size):
        with h5py.File(self.save_path, 'a') as f:
            dset = f[key]
            new_size = dset.shape[0] + append_size
            dset.resize(new_size, axis=0)
            dset[-append_size:] = item

    def __set_items__(self, key):
        dic = self.temporary_dic_write[key]

        if key not in self.temporary_dic_write['keyname']:
            shape = (dic['index'],) if dic['dtype'] == h5py.string_dtype(encoding='utf-8') else np.array(dic['data']).shape
            self.__create_dataset__(key, dic['data'], shape, dic['dtype'])

            if not self.new_file:
                self.__append_dataset__('keyname', key, 1)
            else:
                self.__create_dataset__('keyname', key, (1,), h5py.string_dtype(encoding='utf-8'))
                self.new_file = False

            self.temporary_dic_write['keyname'].append(key)
            self.temporary_dic_read[key] = {
                'index': 0,
                'data': [],
            }
            self.__get_items__(key, 0)

        else:
            self.__append_dataset__(key, dic['data'], dic['index'])

        dic['data'] = []
        dic['index'] = 0

    def __save_all__(self):
        for key in self.temporary_dic_write:
            if key == 'keyname':
                continue
            if self.temporary_dic_write[key]['index'] > 0:
                self.__set_items__(key)

    def save(self, key, item):
        if key == 'keyname':
            raise H5pyDatasetError("Cannot save names.")

        dtype = get_dtype(item)
        if key not in self.temporary_dic_write:
            self.temporary_dic_write[key] = {
                'data':[],
                'index':0,
                'dtype':dtype,
            }
        self.temporary_dic_write[key]['data'].append(item)
        self.temporary_dic_write[key]['index'] += 1

        if self.temporary_dic_write[key]['index'] == self.batch_size:
            self.__set_items__(key)

    def __remove__(self, key):
        self.temporary_dic_write.pop(key)
        with h5py.File(self.save_path, 'a') as f:
            if key in f:
                del f[key]
                self.temporary_dic_write['keyname'].remove(key)

                f['keyname'].resize(len(self.temporary_dic_write['keyname']), axis=0)
                f['keyname'][...] = self.temporary_dic_write['keyname']

    def remove(self, key):
        if key not in self.temporary_dic_write:
            raise H5pyDatasetError("Cannot find key of dataset")
        self.__remove__(key)

    def remove_all(self):
        for key in self.temporary_dic_write['keyname'].copy():
            self.__remove__(key)

    def shape(self):
        self.__save_all__()
        result = {}
        with h5py.File(self.save_path, 'r') as f:
            for key in self.temporary_dic_write['keyname']:
                dataset = f[key]
                result[key] = {
                    'shape':dataset.shape,
                    'dtype':self.temporary_dic_write[key]['dtype'],
                }
        return result

    def __get_items__(self, key, index):
        with h5py.File(self.save_path, 'r') as f:
            dataset = f[key]
            if index >= dataset.shape[0]:
                raise H5pyDatasetError("index out of range")
            max_index = min(index + self.batch_size, dataset.shape[0])
            self.temporary_dic_read[key]['data'] =  dataset[index:max_index]
            self.temporary_dic_read[key]['index'] = index

    def get(self, key, index):
        if key not in self.temporary_dic_read:
            raise H5pyDatasetError("Cannot find key of dataset")

        dic_index = index - self.temporary_dic_read[key]['index']
        if 0 <= dic_index < self.batch_size:
            return self.temporary_dic_read[key]['data'][dic_index]
        else:
            self.__get_items__(key, index)
            return self.get(key, index)

    def get_list(self, key, index):
        if key not in self.temporary_dic_read:
            raise H5pyDatasetError("Cannot find key of dataset")

        if self.temporary_dic_read[key]['index'] != index:
            self.__get_items__(key, index)

        return self.temporary_dic_read[key]['data']

    def get_all(self, key, function, piece_size=True):
        if key not in self.temporary_dic_read:
            raise H5pyDatasetError("Cannot find key of dataset")

        max_index = self.shape()[key]['shape'][0]
        if piece_size:
            for i in tqdm(range(max_index)):
                if i == 0 or i % self.batch_size == 0:
                    self.__get_items__(key, i)
                function(self.temporary_dic_read[key]['data'][i % self.batch_size])
        else:
            count = 0
            while True:
                try:
                    self.__get_items__(key, count * self.batch_size)
                except H5pyDatasetError:
                    break
                function(self.temporary_dic_read[key]['data'])


if __name__ == '__main__':
    path = "C:/SpliceImageTextData/text.h5"
    h5py = H5pyDatafile(path)
    h5py = 0
    # os.remove(path)






