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
        self.matfile = '/media/Data/datasets/image-to-value-regression-using-deep-learning/wiki_crop_aligned/bin_data/wiki_crop_aligned_data.mat'
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

        # image has shape HWC
        image = Image.fromarray(image.squeeze())
        
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
        name (string): Dataset name in ['Full', 'HeadLess', 'HeadLegLess', 'HeadLegArmLess'].
        nb_channels (int): Number of duplicated channels for each depth image.
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

        # image has shape HWC
        if self.nb_channels == 1:
            image = np.expand_dims(image, 0)
        image = np.transpose(image, (1, 0, 2)) # to WHC   
        image = Image.fromarray(image.squeeze())

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
        name (string): Dataset name in ['Full', 'HeadLess', 'HeadLegLess', 'HeadLegArmLess'].
        nb_channels (int): Number of duplicated channels for each depth image.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    
    '''

    def __init__(self, name, nb_channels, train=True, transform=None):
        assert name in ['Full', 'HeadLess', 'HeadLegLess', 'HeadLegArmLess']
        matfile = name + '_ch' + str(nb_channels) + '.mat'
        
        self.nb_channels = nb_channels
        self.matfile = '/media/Data/datasets/image-to-value-regression-using-deep-learning/sportsmen/bin_data/' + matfile
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

        # image is in the format HWC
        image = Image.fromarray(image.squeeze())

        if self.transform is not None:
            image = self.transform(image) # normalize between -1 and 1

        return image, target #, str(label)
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


#%% ---------------------------------------------------------------------------


class RECTANGLES(torch.utils.data.Dataset):
    '''Rectangles Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    
    '''

    def __init__(self, train=True, transform=None):
        self.matfile = '/media/Data/datasets/image-to-value-regression-using-deep-learning/rectangles/bin_data/rectangles.mat'
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

        # image has shape HWC
        image = Image.fromarray(image.squeeze())
        
        if self.transform is not None:
            image = self.transform(image)

        return image, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


#%% ---------------------------------------------------------------------------


class SHAPES(torch.utils.data.Dataset):
    '''Rectangles Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    
    '''

    def __init__(self, train=True, transform=None):
        self.matfile = '/media/Data/datasets/image-to-value-regression-using-deep-learning/shapes/bin_data/shapes.mat'
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

        # image has shape HWC
        image = Image.fromarray(image.squeeze())
        
        if self.transform is not None:
            image = self.transform(image)

        return image, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


#%% ---------------------------------------------------------------------------


class LEGLESS(torch.utils.data.Dataset):
    '''FatNet Dataset.

    Args:
        name (string): Dataset name in ['Full', 'HeadLess', 'HeadLegLess', 'HeadLegArmLess'].
        nb_channels (int): Number of duplicated channels for each depth image.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    
    '''

    def __init__(self, train=True, transform=None):
        
        if train:
            matfile = 'train.mat'
        else:
            matfile = 'test.mat'
            
        self.matfile = '/media/Data/datasets/FatNet/legless/bin_data/' + matfile
        self.transform = transform
        self.train = train  # training set or test set

        data = sio.loadmat(self.matfile)
        images = np.squeeze(data['images'])
        values = np.squeeze(data['targets'])

        if self.train:
            self.train_data = images
            self.train_values = values
        else:
            self.test_data = images
            self.test_values = values

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

        # image is in the format HWC
        image = Image.fromarray(image.squeeze())

        if self.transform is not None:
            image = self.transform(image) # normalize between -1 and 1

        return image, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)