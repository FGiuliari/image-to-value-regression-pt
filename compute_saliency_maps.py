from __future__ import print_function

import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable

from network import Net

import dataset

import os
import numpy as np


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
target_shape = (224, 224)
target_shape = (nb_channels,) + target_shape
batch_size = 1 # <== reduce this value if you encounter memory errors

# the images are normalized between 0 and 1 (thanks to the ToTensor transformation) and then normalized between -1 and +1.
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * nb_channels, (0.5,) * nb_channels)])
test_set = dataset.FATDATA('HeadLegArmLess', nb_channels, train=False, transform=transf)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

model_filename = 'network_state_dict.ckpt'
assert os.path.exists(model_filename)

print('Loading model file', model_filename)
net = Net(target_shape, vgg16_basemodel=False, batch_normalization=True, dropout=False)
net.load_state_dict(torch.load(model_filename))

if HAS_CUDA:
    net.cuda(gpu_id)


def generate_occluded_imgs(image, wnd_size):
    nb_images = image.shape[0] * image.shape[1]
    images = np.zeros((nb_images,) + image.shape, dtype=np.float32)
    coords = np.zeros((nb_images, 2), dtype=np.uint32)
    counter = 0
    h_size = int(wnd_size / 2)
    for u in range(h_size, image.shape[0]-h_size):
        for v in range(h_size, image.shape[1]-h_size):
            tmp = np.array(image, copy=True)
            tmp[u-h_size:u+h_size, v-h_size:v+h_size] = 0  # mask color
            images[counter] = tmp.astype(np.float32)
            coords[counter] = (u, v)
            counter += 1
    images = images[:counter]
    coords = coords[:counter]
    return np.squeeze(images), coords


SMAP = np.zeros((len(test_loader),) + (224, 224), dtype=np.float32)
for i, data in enumerate(test_loader):
    print('Computing GT')
    images, targets = data
    if HAS_CUDA:
        img = images.cuda(gpu_id)
    gt = net(Variable(img))
    gt = gt.data.cpu().squeeze().numpy()[0]
    print('Generating masks...')
    masked, coords = generate_occluded_imgs(images.squeeze().numpy(), 7)
    print('Computing saliency')
    for j in range(masked.shape[0]):
        #img = transf(np.expand_dims(masked[j], 2))
        img = torch.FloatTensor(np.expand_dims(masked[j], 0))
        if HAS_CUDA:
            img = img.cuda(gpu_id)
        yy = net(Variable(img.unsqueeze(0)))
        WERj = yy.data.cpu().squeeze().numpy()[0] - test_set.test_values[i]
        SMAP[i, coords[j][0], coords[j][1]] = WERj
    break
