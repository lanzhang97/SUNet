import numpy as np
import cv2 
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        npimage = npimage.transpose((2, 0, 1))

        First_Label = npmask.copy()
        First_Label[npmask == 1] = 1.
        First_Label[npmask == 2] = 0.
        First_Label[npmask == 3] = 0.
        First_Label[npmask == 4] = 0.
        #print(First_Label.shape)
        Second_Label = npmask.copy()
        Second_Label[npmask == 1] = 0.
        Second_Label[npmask == 2] = 1.
        Second_Label[npmask == 3] = 0.
        Second_Label[npmask == 4] = 0.
        Third_Label = npmask.copy()
        Third_Label[npmask == 1] = 0.
        Third_Label[npmask == 2] = 0.
        Third_Label[npmask == 3] = 1.
        Third_Label[npmask == 4] = 0.
        Fourth_Label = npmask.copy()
        Fourth_Label[npmask == 1] = 0.
        Fourth_Label[npmask == 2] = 0.
        Fourth_Label[npmask == 3] = 0.
        Fourth_Label[npmask == 4] = 1.
        nplabel = np.empty((512, 512, 4))# ((512, 512, 3))
        nplabel[:, :, 0] = First_Label
        nplabel[:, :, 1] = Second_Label
        nplabel[:, :, 2] = Third_Label
        nplabel[:, :, 3] = Fourth_Label
        nplabel = nplabel.transpose((2, 0, 1))#这是一个转置函数

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")

        return npimage,nplabel
