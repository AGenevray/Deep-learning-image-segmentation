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

## Image retrieval from output probabilities ##
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

labels = np.array(['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void'])
label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def print_image(image_pl, width, height, num_classes, labelled_im):
    plt.figure(figsize=(12,4))
    # Plot the obtained image
    plt.subplot(121)
    image_out = np.argmax(image_pl, axis=-1)[0]
    image = np.full([height, width, 3], label_colours[num_classes-1]).astype(np.uint8)
    for c in range(num_classes):
        image[image_out == c] = label_colours[c]
    plt.imshow(image)
    plt.title('Output labels')
    
    # Plot the labelled image
    plt.subplot(122)
    labelled = np.argmax(labelled_im, axis=-1)[0]
    true_labels = np.full([height, width, 3], label_colours[num_classes-1]).astype(np.uint8)
    for c in range(num_classes):
        true_labels[labelled == c] = label_colours[c]
    plt.imshow(true_labels)
    plt.title('Original image labels')
    plt.show()
    
    # Plot the labels legend
    plt.figure(figsize=(12,1))
    for c in range(num_classes):
        plt.subplot(1, num_classes+1, c+1)
        plt.axis('off')
        plt.imshow(np.full([10, 20, 3], label_colours[c]).astype(np.uint8))
        plt.title(labels[c])
    plt.show()
                   
def compute_iou(output, one_hot_labelled, num_classes, width, height):
    iou = np.zeros((num_classes))
    class_label = np.argmax(one_hot_labelled, axis=-1)
    class_output = np.argmax(output, axis=-1)
    # For all the classes except void
    for i in range(output.shape[0]):
        for c in range(num_classes-1):
            lab_i = np.reshape(class_label[i, :, :], width*height)
            out_i = np.reshape(class_output[i, :, :], width*height)
            intersection = np.sum((lab_i == c) & (out_i == c))
            union = np.sum((lab_i == c) | (out_i == c))
            iou[c] = iou[c] + (intersection/union if union != 0 else 1)
    iou = iou/output.shape[1]
    return iou


## Batch generation and data loading ##
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

def random_crop(image, label):
    # 8 pixels to crop in height
    # Random up crop between 0 and 8 pixels
    # Bottom crop will then be 8-(uppder crop) 
    cp = np.random.randint(0, 8);
    return image[cp:-(8-cp), :, :], label[cp:-(8-cp), :]

def load_img_and_labels_from_list(list_filenames):
    img = []
    annot_img = []
    for file in list_filenames:
        image, label = random_crop(imread(file), imread(image_path_to_label_path(file)))
        img.append(image)
        annot_img.append(label)
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
    def __init__(self, data, batch_size=5, num_classes=11,
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
            yield i, x_batch, y_batch

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
            yield i, x_batch, y_batch
            

    def gen_train(self):
        x_batch, y_batch = self._batch_init()
        iteration = 0
        i = 0
        while (iteration < self._num_iterations):
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