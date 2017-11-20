#!/usr/bin/python3


#%% ---------------------------------------------------------------------------
# Imports.


from __future__ import print_function

import torch
from torchvision import transforms
from torch.autograd import Variable

import dataset

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--features', type=str, required=True)
parser.add_argument('--regressor', type=str, required=True)
args = parser.parse_args()

assert args.features in ['hog', 'pca', 'vgg16']
assert args.regressor in ['svm', 'random_forest']


#%% ---------------------------------------------------------------------------
# Feature extraction.


seed = 23092017
np.random.seed(seed)
torch.manual_seed(seed)
    
HAS_CUDA = True    
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False
else:
    torch.cuda.manual_seed(seed)
    gpu_id = 0
    
task = 'shapes' # age-from-faces, gender-from-depth, fat-from-depth, shapes
dsname = 'HeadLegArmLess' # Full, HeadLess, HeadLegLess, HeadLegArmLess
features_to_extract = args.features # hog, pca, vgg16
regressor_mode = args.regressor # svm, random_forest
shuffle_train_set = True

baseline_folder = 'baseline_results/' + task + '/'
if not os.path.exists(baseline_folder):
    os.makedirs(baseline_folder)

feature_filename = baseline_folder + 'features_' + features_to_extract + '.pth'
regressor_filename = baseline_folder + 'regressor_' + regressor_mode + '_using_' + features_to_extract + '.pth'

print('******************')
print('Task:', task)
print('Feature:', features_to_extract)
print('Regression:', regressor_mode)
print('******************')

rows = 181 if task == 'age-from-faces' else 224
cols = 121 if task == 'age-from-faces' else 224

# load extracted features
if os.path.exists(feature_filename):
    
    train_set, test_set = torch.load(feature_filename)
    
else:

    print('Loading dataset...')
    
    # Load dataset.
    nb_channels = 1 if features_to_extract in ['hog', 'pca'] else 3
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * nb_channels, (0.5,) * nb_channels)])

    if task == 'age-from-faces':
        train_set = dataset.FACES(True, transf)
        test_set = dataset.FACES(False, transf)

    if task == 'gender-from-depth':
        train_set = dataset.FATSYNTH(dsname, nb_channels, train=True, transform=transf)
        test_set = dataset.FATSYNTH(dsname, nb_channels, train=False, transform=transf)

    if task == 'fat-from-depth':
        train_set = dataset.FATDATA(dsname, nb_channels, train=True, transform=transf)
        test_set = dataset.FATDATA(dsname, nb_channels, train=False, transform=transf)

    if task == 'shapes':
        train_set = dataset.SHAPES(train=True, transform=transf)
        test_set = dataset.SHAPES(train=False, transform=transf)

    batch_size = 1 if features_to_extract in ['hog', 'pca'] else 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train_set, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


    # Extract features from datset.
    def extract_features(data_loader, network=None):
    
        max_steps = len(data_loader)
        features = torch.FloatTensor()
        targets = torch.FloatTensor()
        
        for i, data in enumerate(data_loader):
            
            x, y = data

            if features_to_extract == 'hog':
                from skimage.feature import hog
                x = x.squeeze().numpy()
                y = y.squeeze()
                if len(x.shape) == 3:
                    x = np.transpose(x, (1, 2, 0))
                    from skimage import data, color
                    x = color.rgb2gray(x)
                batch_features = hog(x, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
                batch_features = torch.FloatTensor(batch_features)
                batch_features = batch_features.view(1, batch_features.numel())

            if features_to_extract == 'pca':
                x = x.squeeze().numpy()
                y = y.squeeze()
                if len(x.shape) == 3:
                    x = np.transpose(x, (1, 2, 0))
                    from skimage import color
                    x = color.rgb2gray(x)
                batch_features = torch.FloatTensor(x.reshape(1, x.size))
            
            if features_to_extract == 'vgg16':
                assert network is not None
                
                if HAS_CUDA:
                    x = x.cuda(gpu_id)
                
                batch_features = network(Variable(x, requires_grad=False))
                batch_features = batch_features.data.cpu()
            
            if features.numel() == 0:
                features = batch_features
                targets = y
            else:
                features = torch.cat((features, batch_features))
                targets = torch.cat((targets, y))
            
            if i % 100 == 0:
                print('[%4d/%4d]' % (i + 1, max_steps))
    
        return features, targets
    
    vgg16 = None
    if features_to_extract == 'vgg16':
        import torchvision.models as models
        vgg16 = models.vgg16_bn(pretrained=True).features
        if HAS_CUDA:
            vgg16.cuda(gpu_id)
        vgg16.eval()
    
    print('Extracting features...')
    train_set = extract_features(train_loader, vgg16)
    test_set = extract_features(test_loader, vgg16)
    
    # Save features.
    print('Saving features...')
    torch.save((train_set, test_set), feature_filename)

if features_to_extract == 'pca':
    from sklearn.decomposition import IncrementalPCA
    print('Reducing feature dimension using Incremental PCA...')
    new_feat_size = 200
    ipca = IncrementalPCA(n_components=new_feat_size, batch_size=new_feat_size)
    ipca.fit(train_set[0].numpy())
    '''eigenfaces = ipca.components_.reshape((new_feat_size, rows, cols, 1))
    plt.figure()
    for i in range(60):
        plt.subplot(6,10,i+1)
        plt.imshow(np.squeeze(eigenfaces[i]))'''
    new_train_data = torch.FloatTensor(ipca.transform(train_set[0].numpy()))
    train_set = (new_train_data, train_set[1])
    new_test_data = torch.FloatTensor(ipca.transform(test_set[0].numpy()))
    test_set = (new_test_data, test_set[1])
    #print('Saving PCA features...')
    #torch.save((train_set, test_set), feature_filename)


#%% ---------------------------------------------------------------------------
# Compute baselines.


x_train, y_train = train_set[0].numpy(), train_set[1].numpy()
x_test, y_test = test_set[0].numpy(), test_set[1].numpy()

if features_to_extract == 'vgg16':
    nb_feats = int(np.prod(x_train.shape[1:]))
    x_train = x_train.reshape((-1, nb_feats))
    x_test = x_test.reshape((-1, nb_feats))


def fit_model(filename, x_train, y_train):
    
    import pickle

    if os.path.exists(filename):
        
        with open(filename, 'rb') as fp:
            regressor = pickle.load(fp)
            
    else:
        
        if regressor_mode == 'svm':
            from sklearn.svm import SVR
            regressor = SVR()
            
        if regressor_mode == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            nb_trees = 100
            regressor = RandomForestRegressor(max_depth=nb_trees, random_state=0)
        
        print('Training baseline regressor...')
        regressor.fit(x_train, y_train)
        
        with open(filename, 'wb') as fp:
            pickle.dump(regressor, fp, pickle.HIGHEST_PROTOCOL)
        
    return regressor


print('Fitting model...')
regressor = fit_model(regressor_filename, x_train, y_train)


#%% ---------------------------------------------------------------------------
# Test baselines.


preds = regressor.predict(x_test).astype(np.float32)

plt.figure()
plt.grid(True)
xx = np.linspace(1, y_test.size, y_test.size)
yy = np.sort(y_test)
plt.plot(xx, yy, color='g', label='gt')
plt.plot(xx, preds, color='b', label='preds')
plt.show()
