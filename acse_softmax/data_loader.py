import numpy as np
import cv2
import os

__all__ = ['load_test_data', 'load_training_data']
labels = ["covid", "lung_opacity", "pneumonia", "normal"]
img_size = 224


def load_training_data(data_dir):
    """
    Load in training data.
    input: data_dir, str, path of data folder.
    output: np.array(data), np.array of the loaded dataset.
    """

    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            if not img.endswith('.png'):
                continue
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # Reshaping images to preferred size
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


def load_test_data(data_dir, train=True):
    """
    Load in test data.
    input:
    data_dir, str, path of data folder.
    output:
    np.array(data), np.array of the loaded dataset.
    file_names, list of names of each images.
    """

    data = []
    file_names = []
    path = data_dir
    class_num = 0
    num = len(os.listdir(path))
    for i in range(num):
        img = 'test_%d.png' % i
        try:
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            # Reshaping images to preferred size
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            data.append([resized_arr, class_num])
            file_names.append(img.replace('.png', ''))
        except Exception as e:
            print(e)
    return np.array(data), file_names
