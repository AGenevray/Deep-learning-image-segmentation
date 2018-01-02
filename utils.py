import numpy as np
import os
import matplotlib.pyplot as plt

def random_flip(image, label):
    do_flip = np.random.randint(0, 2) == 1
    if do_flip:
        image = np.fliplr(image)
        label = np.fliplr(label)
    return image, label

def image_path_to_label_path(image_path):
    image_name = os.path.basename(image_path)
    label_path = os.path.dirname(image_path) + 'annot'
    label_path = os.path.join(label_path, image_name)
    return label_path

## Batch generation and data loading ##
def onehot(label_img, num_classes):
    out = np.zeros((label_img.shape[0], label_img.shape[1], num_classes))
    for c in range(num_classes):
        out[(label_img[:, :] == c), c] = 1
    return out

def compute_iou(output, one_hot_labelled, dataset):
    num_classes = dataset.NUM_CLASSES
    height = dataset.IMAGE_SHAPE[1]
    width = dataset.IMAGE_SHAPE[0]
    
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

def print_image(image_pl, labelled_im, dataset):
    num_classes = dataset.NUM_CLASSES
    height = dataset.IMAGE_SHAPE[1]
    width = dataset.IMAGE_SHAPE[0]
    label_colours = dataset.LABEL_COLOURS
    labels = dataset.LABELS
    
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