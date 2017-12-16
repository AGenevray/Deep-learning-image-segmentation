import warnings
import numpy as np
from skimage.io import imread
from utils import image_path_to_label_path

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
        map[color_encode([r, g, b])] = len(colours)
        colours.append([r, g, b])
        
    void = [224, 224, 192]
    map[color_encode(void)] = len(colours)
    colours.append(void)
    
    return colours, map

LABELS = np.array(['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'screen', 'void'])
LABEL_COLOURS, MAP_COLOURS = np.array(color_map(len(LABELS)-1))

NUM_CLASSES = len(LABELS)
IMAGE_SHAPE = (448, 448, 3)

DATA_FOLDER = 'VOCdevkit'

target_height = IMAGE_SHAPE[1]
target_width = IMAGE_SHAPE[0]

def resize(image, label):
    
    width = len(image[0])
    height = len(image)
    
    if height < target_height:
        diff = (target_height-height)//2
        new_image = np.zeros((target_height, width, 3), dtype=np.dtype('B'))
        new_label = np.zeros((target_height, width), dtype=np.dtype('B'))
        new_image[diff:diff+height] = image
        new_label[diff:diff+height] = label
        image = new_image
        label = new_label
    elif height > target_height:
        diff = (height-target_height)//2
        image = image[diff:-diff, :, :]
        label = label[diff:-diff, :]
        
        
    if width < target_width:
        diff = (target_width-width)//2
        new_image = np.zeros((target_height, target_width, 3), dtype=np.dtype('B'))
        new_label = np.zeros((target_height, target_width), dtype=np.dtype('B'))
        for i in range(len(image)):
            
            for j in range(diff):
                new_image[i, j] = LABEL_COLOURS[0]
                new_label[i, j] = 0
                
            new_image[i, diff:diff+width] = image[i]
            new_label[i, diff:diff+width] = label[i]
            
            for j in range(width + diff, target_width):
                new_image[i, j] = LABEL_COLOURS[0]
                new_label[i, j] = 0
                
        image = new_image
        label = new_label
    elif width > target_width:
        diff = (width-target_width)//2
        image = image[:, diff:-diff, :]
        label = label[:, diff:-diff]
    
    return image, label    
    
def load_img_and_labels_from_list(list_filenames):
    img = []
    annot_img = []
    for file in list_filenames:
        image, label = resize(imread(file), decode_labels(imread(image_path_to_label_path(file))))
        img.append(image)
        annot_img.append(label)
    return img, annot_img

def decode_labels(label):
    new = np.zeros((len(label), len(label[0])), dtype=np.dtype('B'))
    for i in range(len(new)):
        for j in range(len(new[i])):
            new[i, j] = MAP_COLOURS[color_encode(label[i, j].tolist())]
    return new