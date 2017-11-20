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

nb_channels = 3
target_image_res = (224, 224)
use_vgg16_basemodel = True

# the images are normalized between 0 and 1 (thanks to the ToTensor transformation) and then normalized between -1 and +1.
if not use_vgg16_basemodel:
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * nb_channels, (0.5,) * nb_channels)])
else:
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
test_set = dataset.FATDATA_CROP('HeadLegLess', nb_channels, train=False, transform=transf)

# sort test set according to the ground truth value
idx = np.argsort(test_set.test_values)
idx = np.asarray([idx[i] for i in range(0, idx.size, 1)])
test_set.test_values = test_set.test_values[idx]
test_set.test_data = test_set.test_data[idx]

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

nb_samples = len(test_loader)
gt_values = test_set.test_values

model_filename = 'network.pth'
assert os.path.exists(model_filename)

print('Loading model file', model_filename)
net = torch.load(model_filename)
net.eval() # must set the network in evaluation mode (by default, batch normalization and dropout are in training mode)

if HAS_CUDA:
    net.cuda(gpu_id)

results_filename = 'saliency_maps.pth'


def compute_saliency_map(model, tensor_image, k_size, ref_target, thr=0.0):
    assert k_size >= 3 and k_size % 2 == 1

    saliency_values = []
    ch, rows, cols = tensor_image.shape
    h_size = int(k_size / 2)

    for u in range(h_size, rows, h_size):
        for v in range(h_size, cols, h_size):
            masked_image = tensor_image.clone()
            mask_color = torch.mean(masked_image[:, u-h_size:u+h_size, v-h_size:v+h_size])
            masked_image[:, u-h_size:u+h_size, v-h_size:v+h_size] = mask_color
            img = masked_image.unsqueeze(0)
            pred = model(torch.autograd.Variable(img)).cpu().data[0][0]
            pred_err = pred - ref_target
            if np.abs(pred_err) < thr:
                pred_err = 0.0
            saliency_values.append(pred_err)
    
    size = int(np.sqrt(len(saliency_values)))
    saliency_map = np.array(saliency_values, dtype=np.float32)
    saliency_map = saliency_map.reshape((size, size))

    return saliency_map


#%% ---------------------------------------------------------------------------
# Generate saliency maps.

if not os.path.exists(results_filename):
    
    saliency_maps = np.zeros((nb_samples,) + target_image_res, dtype=np.float32)
    '''for i, data in enumerate(test_loader):
        print('[%3d/%3d] %.2f %%' % (i + 1, nb_samples, 100 * (i + 1) / nb_samples))
        image, target = data
        if HAS_CUDA:
            image = image.cuda(gpu_id)
        ref_pred = net(Variable(image)).cpu().data[0][0]
        ch, rows, cols = image[0].shape
        k_size = 15
        h_size = int(k_size / 2)
        for u in range(h_size, rows, h_size):
            for v in range(h_size, cols, h_size):
                masked_image = image[0].clone()
                masked_image[:, u-h_size:u+h_size, v-h_size:v+h_size] = 0  # mask color
                coords = (u, v)
                img = masked_image.unsqueeze(0)
                if HAS_CUDA:
                    img = img.cuda(gpu_id)
                pred = net(Variable(img)).cpu().data[0][0]
                WERj = pred - gt_values[i]
                saliency_maps[i, coords[0], coords[1]] = WERj'''

    from scipy.misc import imresize

    for i, data in enumerate(test_loader):
        print('[%3d/%3d] %.2f %%' % (i + 1, nb_samples, 100 * (i + 1) / nb_samples))
        image, target = data
        if HAS_CUDA:
            image = image.cuda(gpu_id)
        smap = compute_saliency_map(net, image[0], 23, gt_values[i])
        mask = imresize(smap, target_image_res)
        saliency_maps[i] = mask
    
    
    # ---------------------------------------------------------------------------
    # Compute statistics on data.
    
    
    from scipy.ndimage.filters import gaussian_filter
    
    
    smaps_smoothed = np.zeros(saliency_maps.shape, dtype=np.float32)
    smaps_mean = np.zeros(target_image_res, dtype=np.float32)
    smaps_variance = np.zeros(target_image_res, dtype=np.float32)
    
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
    torch.save((info, saliency_maps, smaps_smoothed, smaps_mean, smaps_variance), results_filename)
    
else:
    
    info, saliency_maps, smaps_smoothed, smaps_mean, smaps_variance = torch.load(results_filename)


import matplotlib.pyplot as plt


plt.figure()
plt.title('Mean')
plt.imshow(smaps_mean)
#plt.clim(0, 1)
plt.colorbar()

plt.figure()
plt.title('Variance')
plt.imshow(smaps_variance)
#plt.clim(0, 1)
plt.colorbar()

plt.show()


#%% ---------------------------------------------------------------------------
# Show saliency video.


smaps = smaps_smoothed - np.min(smaps_smoothed)
smaps /= np.max(smaps)


# show saliency "video"
plt.figure()
for i in range(nb_samples):
    plt.clf() # speed up the computation avoiding hold-on behavior
    plt.title('Saliency maps: ' + str(i + 1) + ' gt: ' + str(gt_values[i]))
    plt.imshow(smaps[i], cmap=plt.get_cmap('jet'))
    #plt.clim(0, 1)
    plt.colorbar()
    plt.pause(0.250)


#%% ---------------------------------------------------------------------------
# Compute bins.


bin_thr = 1.8
bins = []
cur_bin = []
min_gt_of_current_bin = gt_values[0]

for i in range(nb_samples):
    if gt_values[i] - min_gt_of_current_bin <= bin_thr:
        cur_bin.append(i)
    else:
        bins.append(cur_bin)
        min_gt_of_current_bin = gt_values[i]
        cur_bin = [i]

bins.append(cur_bin)
nb_bins = len(bins)
print('#bins:', nb_bins)


plt.figure()
plt.grid(True)
#plt.plot(np.linspace(0, nb_samples, nb_samples), gt_values)
for b in bins:
    plt.plot(np.linspace(b[0], b[-1], len(b)), gt_values[b])
plt.show()


#%% ---------------------------------------------------------------------------
# Show bin means.


bin_mean = np.zeros((nb_bins,) + target_image_res, dtype=np.float32)
for i, b in enumerate(bins):
    maps = smaps_smoothed[b]
    bmean = np.zeros(target_image_res, dtype=np.float32)
    for m in maps:
        bmean += m
    bin_mean[i] = bmean / maps.shape[0]

# show bin-saliency "video"
plt.figure()
for i in range(nb_bins):
    plt.clf() # speed up the computation avoiding hold-on behavior
    plt.title('Bin: ' + str(i + 1))
    plt.imshow(bin_mean[i], cmap=plt.get_cmap('jet'))
    #plt.clim(0, 1)
    plt.colorbar()
    plt.pause(0.80)


#%% ---------------------------------------------------------------------------
# Show bin volume.


from mpl_toolkits.mplot3d import Axes3D


margin = 18
xs = np.linspace(margin, target_image_res[0] - margin, target_image_res[0] - margin*2)
ys = np.linspace(margin, target_image_res[1] - margin, target_image_res[1] - margin*2)
XS, YS = np.meshgrid(ys, xs)

fig = plt.figure()
for i in range(bin_mean.shape[0]):
    print(i + 1, bin_mean.shape[0])
    plt.clf()
    plt.title(str(i))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(XS,YS,bin_mean[i, margin:target_image_res[0]-margin, margin:target_image_res[1]-margin])
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_zlabel('score')
    #ax.set_zlim(0.0, 0.06)
    plt.pause(1.0)

#plt.show()
