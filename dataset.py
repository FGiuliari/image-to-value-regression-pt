from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
from PIL import Image
import numpy as np


class FACES(torch.utils.data.Dataset):
    '''Faces Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    
    '''

    def __init__(self, train=True, transform=None):
        self.matfile = '/media/Data/datasets/wiki_crop_aligned/bin_data/wiki_crop_aligned_data.mat'
        self.transform = transform
        self.train = train  # training set or test set

        data = sio.loadmat(self.matfile)
        images = np.squeeze(data['images'])
        values = np.squeeze(data['values'])
        train_idx = np.squeeze(data['train_idx'])
        test_idx = np.squeeze(data['test_idx'])

        if self.train:
            self.train_data = images[train_idx]
            self.train_values = values[train_idx]
        else:
            self.test_data = images[test_idx]
            self.test_values = values[test_idx]

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index.

        Returns:
            tuple: (image, target).
        '''
        if self.train:
            image, target = self.train_data[index], self.train_values[index]
        else:
            image, target = self.test_data[index], self.test_values[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = np.swapaxes(image, 0, 1) # return to PIL format W*H
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

