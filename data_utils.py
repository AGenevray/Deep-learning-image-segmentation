import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import datetime
from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.preprocessing import LabelEncoder
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.cross_validation import StratifiedShuffleSplit

import os
import subprocess
import itertools

def onehot(label_img, num_classes):
    out = np.zeros((label_img.shape[0], label_img.shape[1], num_classes))
    for c in range(num_classes):
        out[(label_img[:, :] == c), c] = 1
    return out

def image_path_to_label_path(image_path):
    image_name = os.path.basename(image_path)
    label_path = os.path.dirname(image_path) + 'annot'
    label_path = os.path.join(label_path, image_name)
    return label_path

def load_img_and_labels_from_list(list_filenames):
    img = []
    annot_img = []
    for file in list_filenames:
        img.append(imread(file))
        annot_img.append(imread(image_path_to_label_path(file)))
    return img, annot_img

class load_data():      
    def __init__(self, images_path):
        train_path = os.path.join(images_path, 'train', '*')
        valid_path = os.path.join(images_path, 'val', '*')
        test_path = os.path.join(images_path, 'test', '*')
        self._load(train_path, valid_path, test_path)
    
    def _load(self, train_path, valid_path, test_path):
        train_files = glob.glob(train_path)
        test_files = glob.glob(test_path)
        valid_files = glob.glob(valid_path)
        self.train, self.train_label = load_img_and_labels_from_list(train_files)
        self.test, self.test_label = load_img_and_labels_from_list(test_files)
        self.valid, self.valid_label = load_img_and_labels_from_list(valid_files)
        
        
class batch_generator():
    def __init__(self, data, batch_size=64, num_classes=12,
                 num_iterations=5e3, num_features=64, seed=42, val_size=0.1):
        self._train = data.train
        self._test = data.test
        self._valid = data.valid
        self._train_label = data.train_label
        self._test_label = data.test_label
        self._valid_label = data.valid_label
        # get image size
        value = self._train[0]
        self._image_shape = list(value.shape)
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._num_iterations = num_iterations
        self._num_features = num_features
        self._seed = seed
        self._val_size = 0.1
        self._idcs_train = list(range(len(data.train)))
        self._idcs_valid = list(range(len(data.valid)))
        self._idcs_test = list(range(len(data.test)))
        
    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def _batch_init(self):
        x_batch_holder = np.zeros(tuple([self._batch_size] + self._image_shape), dtype='int')
        y_batch_holder = np.zeros(tuple([self._batch_size] + [self._image_shape[0]] + [self._image_shape[1]] + [self._num_classes]), dtype='int')
        return x_batch_holder, y_batch_holder

    def gen_valid(self):
        x_batch, y_batch = self._batch_init()
        i = 0
        for idx in self._idcs_valid:
            x_batch[i] = self._valid[idx]
            y_batch[i] = onehot(self._valid_label[idx], self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield i, x_batch, y_batch
                x_batch, y_batch = self._batch_init()
                i = 0
        if i != 0:
            yield x_batch, y_batch

    def gen_test(self):
        x_batch, y_batch = self._batch_init()
        i = 0
        for idx in self._idcs_test:
            x_batch[i] = self._test[idx]
            y_batch[i] = onehot(self._test_label[idx], self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield i, x_batch, y_batch
                x_batch, y_batch = self._batch_init()
                i = 0
        if i != 0:
            yield x_batch, y_batch
            

    def gen_train(self):
        x_batch, y_batch = self._batch_init()
        iteration = 0
        i = 0
        while True:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                # extract data from dict
                x_batch[i] = self._train[idx]
                y_batch[i] = onehot(self._train_label[idx], self._num_classes)
                i += 1
                if i >= self._batch_size:
                    yield x_batch, y_batch
                    x_batch, y_batch = self._batch_init()
                    i = 0
                    iteration += 1
                    if iteration >= self._num_iterations:
                        break