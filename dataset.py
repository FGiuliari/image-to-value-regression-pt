from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
from PIL import Image
import numpy as np


#%% ---------------------------------------------------------------------------


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


#%% ---------------------------------------------------------------------------


class FATSYNTH(torch.utils.data.Dataset):
    '''Synthetic fat Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    
    '''

    def __init__(self, name, nb_channels, train=True, transform=None):

        assert name in ['Full', 'HeadLess', 'HeadLegLess', 'HeadLegArmLess']
        matfile = name + '_ch' + str(nb_channels) + '_synth.mat'

        self.nb_channels = nb_channels
        self.matfile = '/media/Data/datasets/FatNet/synthdata/bin_data/' + matfile
        self.transform = transform
        self.train = train  # training set or test set

        data = sio.loadmat(self.matfile)

        if self.train:
            self.train_data = np.squeeze(data['train_images'])
            self.train_values = np.squeeze(data['train_values'])
        else:
            self.test_data = np.squeeze(data['test_images'])
            self.test_values = np.squeeze(data['test_values'])

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
        image = np.swapaxes(image, 0, 1)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        '''
        # add random noise
        noise_scale = 0.5
        image += 2 * (torch.rand(image.size()) - 0.5) * (1 - noise_scale)

        def where(cond, x1, x2):
            return (cond.float() * x1) + ((1 - cond.float()) * x2)

        image = where(image < -1, -torch.ones(image.size()), image)
        image = where(image > 1, torch.ones(image.size()), image)
        '''

        return image, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


#%% ---------------------------------------------------------------------------


class FATDATA(torch.utils.data.Dataset):
    '''FatNet Dataset.

    Args:
        name (string): Dataset name [FullR, HeadLegArmlessR, HeadLeglessR, HeadlessR].
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    
    '''

    def __init__(self, name, train=True, transform=None):
        assert name in ['Full_ch1', 'HeadLegArmLess_ch1', 'HeadLegLess_ch1', 'HeadLess_ch1',
                        'Full_ch3', 'HeadLegArmLess_ch3', 'HeadLegLess_ch3', 'HeadLess_ch3']
        self.matfile = '/media/Data/datasets/image-to-value-regression-using-deep-learning/sportsmen/bin_data/' + name + '.mat'
        self.transform = transform
        self.train = train  # training set or test set

        data = sio.loadmat(self.matfile)
        images = np.squeeze(data['images'])
        labels = np.squeeze(data['labels'])
        values = np.squeeze(data['values'])
        train_idx = np.squeeze(data['train_idx'])
        test_idx = np.squeeze(data['test_idx'])

        if self.train:
            self.train_data = images[train_idx]
            self.train_labels = labels[train_idx]
            self.train_values = values[train_idx]
        else:
            self.test_data = images[test_idx]
            self.test_labels = labels[test_idx]
            self.test_values = values[test_idx]

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index.

        Returns:
            tuple: (image, target).
        '''
        if self.train:
            image, target, label = self.train_data[index], self.train_values[index], self.train_labels[index]
        else:
            image, target, label = self.test_data[index], self.test_values[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target, str(label)
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)