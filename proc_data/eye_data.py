"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu
--------------------------
Code written by Yuezun Li
"""
import cPickle, os
import numpy as np
from pathlib2 import Path
import cv2, sys
sys.path.append('..')
import py_utils.img_utils.proc_img as ulib


class EyeData(object):

    def __init__(self,
                 anno_path,
                 data_dir,
                 batch_size,
                 is_augment,
                 is_shuffle,
                 ):
        self.anno_path = anno_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.is_augment = is_augment
        self.is_shuffle = is_shuffle

        # Load annotations
        with open(self.anno_path, 'rb') as f:
            self.annos = cPickle.load(f)
        # Get data number
        self.data_num = len(self.annos)
        if self.is_shuffle:
            self._shuffle()

        self.batch_num = np.int32(np.ceil(self.data_num / self.batch_size))

    def _shuffle(self):
        random_idx = np.random.permutation(self.data_num)
        tmp = [self.annos[i] for i in random_idx]
        self.annos = tmp

    def get_batch(self, batch_idx, size=None):
        if batch_idx >= self.batch_num:
            raise ValueError("Batch idx must be in range [0, {}].".format(self.batch_num - 1))

        # Get start and end image index ( counting from 0 )
        start_idx = batch_idx * self.batch_size
        idx_range = []
        for i in range(self.batch_size):
            idx_range.append((start_idx + i) % self.data_num)

        print('batch index: {}, counting from 0'.format(batch_idx))

        img_tensor = []
        label_list = []
        im_name_list = []
        for i in idx_range:
            # seq name
            im_name, label = self.annos[i]
            im_dir = Path(self.data_dir) / (im_name + '.png')
            im = cv2.imread(str(im_dir))
            # Is augment?
            if self.is_augment:
                im = ulib.aug([im], color_rng=[0.8, 1.2])[0]
            # resize
            if size is not None:
                # Padding and resize to out
                im = cv2.resize(im, tuple(size))
            img_tensor.append(im)
            label_list.append(label)
            im_name_list.append(im_name)

        img_tensor = np.array(img_tensor)
        label_list = np.array(label_list)

        if batch_idx == self.batch_num:
            if self.is_shuffle:
                self._shuffle()
        return img_tensor, label_list, im_name_list