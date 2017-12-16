import numpy as np
import glob
import warnings
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import os

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_bit(i, bit_number):
    mask = 1 << bit_number
    return (i & mask) >> bit_number


def color_encode(color):
    return ','.join([str(x) for x in color])

def color_map(N):
    N = N * 2
    colours = []
    map = {}
    for i in range(1, N+1, 2):
        id = i - 1
        r = 0
        g = 0
        b = 0
        for j in range(8):
            r = r | (get_bit(id, 1) << 7 - j)
            g = g | (get_bit(id, 2) << 7 - j)
            b = b | (get_bit(id, 3) << 7 - j)
            id = id >> 3
        colours.append([r, g, b])
        map[color_encode([r, g, b])] = len(colours)
        
    void = [224, 224, 192]
    colours.append(void)
    map[color_encode(void)] = 0
    
    return colours, map

def split_images():
    val_prop = 0.1
    train_prop = 0.7
    
    dir = os.path.join('data', 'VOCdevkit')
    labels_dir = os.path.join(dir, 'default', 'label')
    images_dir = os.path.join(dir, 'default', 'image')
    list_images = glob.glob(os.path.join(dir, 'default', 'label', '*'))
    num_images = len(list_images)
    
    paths = [('val', 'valannot'), ('train', 'trainannot'), ('test', 'testannot')]
    
    for id_image in range(num_images):
        
        random = np.random.random_sample()
        if random < val_prop:
            id = 0
        elif random < val_prop + train_prop:
            id = 1
        else:
            id = 2
        
        image_name = os.path.split(list_images[id_image])[-1]
        print(image_name)
        
        os.rename(os.path.join(labels_dir, image_name), os.path.join(dir, paths[id][1], image_name))
        os.rename(os.path.join(images_dir, image_name[:-3] + 'jpg'), os.path.join(dir, paths[id][0], image_name))

labels = np.array(['void', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'screen'])
label_colours, map_colours = np.array(color_map(len(labels)))
print(label_colours)

target_height = 448
target_width = 448

def random_crop(image, label):
    width = len(image[0])
    height = len(image)
    
    if height < target_height:
        new_image = np.zeros((target_height, width, 3), dtype=np.dtype('B'))
        new_label = np.zeros((target_height, width), dtype=np.dtype('B'))
        for i in range(height):
            new_image[i+(target_height-height)//2] = image[i]
            new_label[i+(target_height-height)//2] = label[i]
        image = new_image
        label = new_label
    elif height > target_height:
        image = image[(height-target_height)//2:(target_height-height)//2, :, :]
        label = label[(height-target_height)//2:(target_height-height)//2, :]
        
    if width < target_width:
        new_image = np.zeros((target_height, target_width, 3), dtype=np.dtype('B'))
        new_label = np.zeros((target_height, target_width), dtype=np.dtype('B'))
        for i in range(len(image)):
            for j in range((target_width-width)//2):
                new_image[i, j] = label_colours[0]
                new_label[i, j] = 0
            for j in range(width):
                new_image[i, (target_width-width)//2+j] = image[i, j]
                new_label[i, (target_width-width)//2+j] = label[i, j]
            for j in range(width + (target_width-width)//2, target_width):
                new_image[i, j] = label_colours[0]
                new_label[i, j] = 0
        image = new_image
        label = new_label
    elif width > target_width:
        image = image[:, (width-target_width)//2:(target_width-width)//2, :]
        label = label[:, (width-target_width)//2:(target_width-width)//2]
    
    return image, label    
    
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
    iou = iou/output.shape[0]
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

def load_img_and_labels_from_list(list_filenames):
    img = []
    annot_img = []
    for counter, file in enumerate(list_filenames):
        print("Load", file, counter)
        image, label = random_crop(imread(file), decode_labels(imread(image_path_to_label_path(file))))
        img.append(image)
        annot_img.append(label)
    return img, annot_img

def decode_labels(label):
    new = np.zeros((len(label), len(label[0])), dtype=np.dtype('B'))
    for i in range(len(new)):
        for j in range(len(new[i])):
            new[i, j] = map_colours[color_encode(label[i, j].tolist())]
    return new

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