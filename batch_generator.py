import numpy as np
from utils import onehot, random_flip

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
        while iteration < self._num_iterations:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                # extract data from dict
                x_batch[i], y_batch[i] = random_flip(self._train[idx], onehot(self._train_label[idx], self._num_classes))
                i += 1
                if i >= self._batch_size:
                    yield x_batch, y_batch
                    x_batch, y_batch = self._batch_init()
                    i = 0
            iteration += 1