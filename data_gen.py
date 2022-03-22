import os
import cv2
import pickle
import gc
import math

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class DataGenerator(Sequence):
    
    def __init__(self, x_path, y_path = None, to_fit = True,  seq_len = 3, batch_size = 5):
        gc.collect()
        self.x_path = x_path        
        self.batch_size = batch_size
        self.to_fit = to_fit
        self.list_X = os.listdir(self.x_path)
        self.seq_len = seq_len
        if to_fit:
            self.y_path = y_path
            self.dict_Y = self.get_y(y_path)
    
    
    def __len__(self):
        # return len(self.list_X)
        return math.ceil(len(self.list_X)/self.batch_size)
    
    
    def __getitem__(self, index):
        # print('mmmm')
        gc.collect()
        images_folder = self.list_X[index]
        print(images_folder)
        images_list = sorted(os.listdir(self.x_path + images_folder))
        all_frames = []
        print('mmmm')
        for img in images_list:
            x = np.array(cv2.imread(self.x_path+images_folder+'/'+img, 0)).astype(np.float16)
            x = x.reshape(x.shape + (1,))
            all_frames.append(x)
        print('mmmm1111')
        all_frames = np.stack(all_frames).astype(np.float16)
        print(all_frames[0].shape)
        # all_frames = np.stack(all_frames)
        print('1111')
        key = images_folder.split('_')[:2]
        key = '_'.join(key)
        Y = np.array(self.dict_Y[key])
        print('222222')
        all_frames, targets = self.check(all_frames, Y)
        print('333333')
        series_data = TimeseriesGenerator(all_frames, targets, length = self.seq_len, batch_size=self.batch_size)
        print('mmmm2222222')
        return series_data
    
    def get_y(self, path):
        gc.collect()
        with open(path, 'rb') as pickle_file:
            y_dict = pickle.load(pickle_file)
        return y_dict 
    
    def check(self, all_frames, targets):
        print((len(all_frames), len(targets)))
        gc.collect()
        if all_frames.shape[0] < targets.shape[0]:
            targets = targets[:-1]
        print((len(all_frames), len(targets)))
        return all_frames, targets
