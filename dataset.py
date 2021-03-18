from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import cv2

from data_mean import get_mean

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def load_data(path, partition, real, data_mean):
    dataset = []
    for clip_window in glob.glob(os.path.join(path, 'clip_window_*')):
        img0_path = clip_window + '/frame_00.jpg'
        img1_path = clip_window + '/frame_01.jpg'
        dataset.append(process_data(img0_path, real, data_mean) +
                       process_data(img1_path, real, data_mean))
    return dataset

def process_data(img_path, real, data_mean):
    if real:
        image1 = cv2.imread(img_path).astype(np.float32)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image2 = np.zeros(image1.shape).astype(np.float32)
    else:
        # color image1
        image1 = cv2.imread(img_path).astype(np.float32)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1[:,:,0] += data_mean[2]
        image1[:,:,1] += data_mean[1]
        image1[:,:,2] += data_mean[0]

        # gray image2
        image2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image2 = np.asarray([image2,image2,image2])
        image2= np.transpose(image2,(1,2,0))
    return image1 / 127.5 - 1, image2 / 127.5 - 1

class ImageDataset(Dataset):
    def __init__(self, path, partition='train', real=True, data_mean=None):
        self.path = path
        self.partition = partition
        self.real = real
        self.data_mean = data_mean
        self.dataset = load_data(self.path, self.partition, self.real, self.data_mean)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]
        return image

if __name__ == '__main__':
    data_mean = get_mean('dataset' + '/anime')
    anime_style = ImageDataset('dataset' + '/anime/style', real=False, data_mean=data_mean)
    anime_smooth = ImageDataset('dataset' + '/anime/smooth', real=False, data_mean=data_mean)
    real = ImageDataset('dataset' + '/real')
    test_data = ImageDataset('dataset' + '/test', partition='test')
    print(len(anime_style))
    print(len(anime_smooth))
    print(len(real))
    print(len(test_data))

    import random

    for i in range(5):
        img0, img1 = random.choice(anime_style)
        plt.imshow(((img0 + 1) * 127.5).clip(0, 255).astype(np.uint8))
        plt.waitforbuttonpress()
        plt.imshow(((img1 + 1) * 127.5).clip(0, 255).astype(np.uint8))
        plt.waitforbuttonpress()

    for i in range(5):
        img, _ = random.choice(real)
        plt.imshow(((img + 1) * 127.5).clip(0, 255).astype(np.uint8))
        plt.waitforbuttonpress()
