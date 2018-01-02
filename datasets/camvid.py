import warnings
import numpy as np
from skimage.io import imread
from utils import image_path_to_label_path

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

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

LABELS = np.array(['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void'])
LABEL_COLOURS = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

NUM_CLASSES = len(LABELS)
WEIGHTS = [1 for l in LABELS]
WEIGHTS[-1] = 0
IMAGE_SHAPE = (480, 352, 3)

DATA_FOLDER = 'CamVid'

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

