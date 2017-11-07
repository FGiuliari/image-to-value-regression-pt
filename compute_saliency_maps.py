#!/usr/bin/python3


#%% ---------------------------------------------------------------------------
# Imports.


from __future__ import print_function

import torch
from torchvision import transforms
from torch.autograd import Variable

from network import Net

import dataset

import os
import numpy as np


#%% ---------------------------------------------------------------------------
# Settings.


# set the seed for debugging purposes
seed = 23092017
np.random.seed(seed)
torch.manual_seed(seed)

# check cuda core
HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False
else:
    torch.cuda.manual_seed(seed)
    gpu_id = 0

print('Loading data...')

nb_channels = 1
target_shape = (nb_channels,) + (224, 224)

# the images are normalized between 0 and 1 (thanks to the ToTensor transformation) and then normalized between -1 and +1.
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * nb_channels, (0.5,) * nb_channels)])
test_set = dataset.FATDATA('HeadLegArmLess', nb_channels, train=False, transform=transf)

# sort test set according to the ground truth value
idx = np.argsort(test_set.test_values)
test_set.test_values = test_set.test_values[idx]
test_set.test_data = test_set.test_data[idx]

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

nb_samples = len(test_loader)

model_filename = 'network_state_dict.ckpt'
assert os.path.exists(model_filename)

print('Loading model file', model_filename)
net = Net(target_shape, vgg16_basemodel=False, batch_normalization=True, dropout=False)
net.load_state_dict(torch.load(model_filename))
net.eval() # must set the network in evaluation mode (by default, batch normalization and dropout are in training mode)

if HAS_CUDA:
    net.cuda(gpu_id)


#%% ---------------------------------------------------------------------------
# Generate saliency maps.


def generate_occluded_images(image, k_size):
    ch, rows, cols = image.shape
    nb_pixels = rows * cols
    masked_images = torch.zeros(nb_pixels, ch, rows, cols)
    coords = np.zeros((nb_pixels, 2), dtype=np.uint32)
    it = 0
    h_size = int(k_size / 2)
    for u in range(h_size, rows, h_size):
        for v in range(h_size, cols, h_size):
            masked_images[it] = image.clone()
            masked_images[it][:, u-h_size:u+h_size, v-h_size:v+h_size] = 0  # mask color
            coords[it] = (u, v)
            it += 1
    masked_images = masked_images[:it]
    coords = coords[:it]
    return masked_images, coords


saliency_maps = np.zeros((nb_samples,) + target_shape[1:], dtype=np.float32)
for i, data in enumerate(test_loader):
    print('[%3d/%3d] %.2f %%' % (i + 1, nb_samples, 100 * (i + 1) / nb_samples))
    image, target = data
    if HAS_CUDA:
        image = image.cuda(gpu_id)
    ref_pred = net(Variable(image)).cpu().data[0][0]
    masked_images, coords = generate_occluded_images(image[0], k_size=15)
    nb_masked_images = masked_images.shape[0]
    for j in range(0, nb_masked_images):
        img = masked_images[j].unsqueeze(0)
        if HAS_CUDA:
            img = img.cuda(gpu_id)
        pred = net(Variable(img)).cpu().data[0][0]
        WERj = pred - test_set.test_values[i]
        saliency_maps[i, coords[j][0], coords[j][1]] = WERj


#%% ---------------------------------------------------------------------------


from scipy.ndimage.filters import gaussian_filter


smaps_smoothed = np.zeros(saliency_maps.shape, dtype=np.float32)
smaps_mean = np.zeros((224,224), dtype=np.float32)
smaps_variance = np.zeros((224,224), dtype=np.float32)

print('Computing smoothed saliency for visualization purposes')
for i in range(nb_samples):
    smaps_smoothed[i] = gaussian_filter(saliency_maps[i], 7)

print('Computing mean saliency')
for i in range(nb_samples):
    smaps_mean += smaps_smoothed[i]
    
smaps_mean = smaps_mean / nb_samples

print('Computing saliency variance')
for i in range(nb_samples):
    smaps_variance += np.power(smaps_smoothed[i] - smaps_mean, 2)
    
smaps_variance = smaps_variance / nb_samples

info = 'info, saliency_maps, smaps_smoothed, smaps_mean, smaps_variance'
torch.save((info, saliency_maps, smaps_smoothed, smaps_mean, smaps_variance), 'saliency_maps.pth')


import matplotlib.pyplot as plt


plt.figure()
plt.title('Mean')
plt.imshow(smaps_mean)

plt.figure()
plt.title('Variance')
plt.imshow(smaps_variance)

plt.show()


#%% ---------------------------------------------------------------------------


# show saliency "video"
plt.figure()
for i in range(0, nb_samples, 9):
    gt = test_set.test_values[i]
    plt.title('Saliency maps: ' + str(i + 1) + ' gt: ' + str(gt))
    plt.imshow(smaps_smoothed[i])
    plt.pause(0.250)




