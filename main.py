#%% ---------------------------------------------------------------------------
# Imports.


from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import dataset


HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

seed = 23092017
torch.manual_seed(seed)
if HAS_CUDA:
    torch.cuda.manual_seed(seed)

if HAS_CUDA:
    gpu_id = 0


# ---------------------------------------------------------------------------
# Load dataset.


task = 'fat-from-depth' # age-from-faces, gender-from-depth, fat-from-depth

nb_epochs = 60 # max number of training epochs
batch_size = 24 # <== reduce this value if you encounter memory errors
shuffle_train_set = True
use_batch_norm = True
use_dropout = False
use_vgg16_basemodel = False
use_data_augmentation_hflip = True # WARNING - data augmentation doubles the batch size

nb_channels = 3 if use_vgg16_basemodel or task == 'age-from-faces' else 1 # vgg16 requires RGB images
target_shape = (180, 120) if task == 'age-from-faces' else (224, 224)
target_shape = (nb_channels,) + target_shape

# the images are normalized between 0 and 1 (thanks to the ToTensor transformation) and then normalized between -1 and +1.
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * nb_channels, (0.5,) * nb_channels)])

if task == 'age-from-faces':
    train_set = dataset.FACES(True, transf)
    test_set = dataset.FACES(False, transf)

if task == 'gender-from-depth':
    train_set = dataset.FATSYNTH('HeadLegArmLess', nb_channels, train=True, transform=transf)
    test_set = dataset.FATSYNTH('HeadLegArmLess', nb_channels, train=False, transform=transf)

if task == 'fat-from-depth':
    train_set = dataset.FATDATA('HeadLegArmLess', nb_channels, train=True, transform=transf)
    test_set = dataset.FATDATA('HeadLegArmLess', nb_channels, train=False, transform=transf)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train_set, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Show some samples.


import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5 # unnomalize
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0))) # channel last


# extract some sample images
dataiter = iter(train_loader) # first item is the iterator
images, targets = dataiter.next() # extract batch

plt.figure()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%.3f' % targets[j] for j in range(batch_size)))


def show_stats(x, title):
    plt.figure()
    plt.title(title)
    plt.hist(x, bins='auto')
    print(title)
    print('> shape:', x.shape)
    print('> [mean, var]:', np.mean(x), np.var(x))
    print('> [min, max]:', np.min(x), np.max(x))


show_stats(train_set.train_values, 'TRAIN value distribution')
show_stats(test_set.test_values, 'TEST value distribution')

plt.show()


#%% ---------------------------------------------------------------------------
# Setup network.


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# to build a network, extend the class nn.Module
class Net(nn.Module):
    
    def __init__(self, input_shape, vgg16_basemodel=True, batch_normalization=False, dropout=False):
        # define the network components but not the actual architecture
        super(Net, self).__init__()

        self.use_base_model = vgg16_basemodel
        self.use_batch_normalization = batch_normalization
        self.use_dropout = dropout
        
        if self.use_base_model:
            if self.use_batch_normalization:
                self.base_model = models.vgg16_bn(pretrained=True).features
            else:
                self.base_model = models.vgg16(pretrained=True).features
            for param in self.base_model.parameters():
                self.base_model.requires_grad = True
        else:
            self.pool = nn.MaxPool2d(2, 2)
            self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
            self.conv1_bn = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.conv2_bn = nn.BatchNorm2d(64)

        # to compute the number of vectorized features we need to compute
        # once the forward pass on the feature_extractor
        x = self._features(Variable(torch.zeros(1, *input_shape)))
        self.nfts = x.numel()
        
        self.fc1 = nn.Linear(self.nfts, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
    
    def _features(self, x):
        if self.use_base_model:
            x = self.base_model(x)
        else:
            if self.use_batch_normalization:
                x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
                x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
            else:
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
        return x
    
    def _regressor(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = F.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = F.dropout(x)
        x = self.fc4(x)
        return x
    
    # compute at runtime the forward pass
    def forward(self, x):
        x = self._features(x)
        # "view" is a cost-free function (as "reshape" in numpy)
        x = x.view(-1, self.nfts)
        x = self._regressor(x)
        return x
        

def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.01)
        module.bias.data.fill_(0.0)
    if classname.find('BatchNorm') != -1:
        module.weight.data.normal_(0.0, 0.01)
        module.bias.data.fill_(0.0)
    if classname.find('Linear') != -1:
        module.weight.data.normal_(0.0, 0.01)
        module.bias.data.fill_(0.0)


net = Net(input_shape=target_shape, vgg16_basemodel=use_vgg16_basemodel, batch_normalization=use_batch_norm, dropout=use_dropout)
net.apply(weights_init) # applica per ogni modulo la funzione passata come parametro
print(net)


if HAS_CUDA:
    # all the tensors in the module are converted to cuda
    net.cuda(gpu_id)


#%% ---------------------------------------------------------------------------
# Training.


import torch.optim as optim


criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
#optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0005)


def lr_scheduler(optimizer, epoch=None, lr_decay=0.001, step=1):
    """Decay learning rate by a factor of lr_decay every step epochs.
    """
    if epoch is None or step == 1 or (epoch+1) % step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (1 - lr_decay)
    return optimizer


import os


def predict(network, dataset, batch_size=8):
    network.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    predictions = None
    for data in loader:
        images, _ = data
        if HAS_CUDA:
            images = images.cuda(gpu_id)
        outputs = network(Variable(images))
        outputs = outputs.data.cpu() # free gpu memory
        if predictions is None:
            predictions = outputs
        else:
            predictions = torch.cat((predictions, outputs))
    return predictions


def evaluate(predictions, ground_truth_values):
    loss = criterion(Variable(predictions, requires_grad=False), Variable(ground_truth_values, requires_grad=False))
    residuals = predictions - ground_truth_values
    abs_error = torch.abs(residuals)
    return loss.data[0], torch.mean(abs_error)


model_name = 'model.ckpt'
if os.path.exists(model_name):

    print('Found pretrained model. Loading file', model_name)
    net.load_state_dict(torch.load(model_name))

else:

    best_train_loss = float('Inf')
    best_test_loss = float('Inf')
    best_epoch_id = 0
    valid_training = True
    loss_history = None

    epochs_without_improvement = 0
    max_epochs_without_improvement = 5

    # training loop
    for epoch in range(nb_epochs):

        # early stop check
        if not valid_training:
            break

        # at each epoch, update the optimizer learning rate
        optimizer = lr_scheduler(optimizer, lr_decay=0.25)

        # enable training mode (the function eval() freezes the weight gradients)
        net.train()

        # variables used to print the training progress
        running_loss = 0.0
        max_steps = len(train_loader)
        step = int(max_steps / 5)

        # batch loop
        for i, data in enumerate(train_loader):
            inputs, targets = data
            
            if use_data_augmentation_hflip:
                hflip = torch.from_numpy(inputs.numpy()[:,:,:,::-1].copy())
                inputs = torch.cat((inputs, hflip))
                targets = torch.cat((targets, targets))
            
            # for each batch, update the learning rate
            optimizer = lr_scheduler(optimizer, lr_decay=0.0005)

            # since the network is sent to the GPU, also the input tensors
            # must be sent to the graphics card
            if HAS_CUDA:
                inputs, targets = inputs.cuda(gpu_id), targets.cuda(gpu_id)

            # gradients are accumulated by default, so we need to reset them at each
            # iteration, so that each batch gradient is computed indepentently
            optimizer.zero_grad()

            # convert the input tensor to a Variable (autograd wrapper with gradient
            # utilities)
            outputs = net(Variable(inputs))

            # compute the loss according to the desired criterion
            loss = criterion(outputs, Variable(targets))

            if loss.data[0] == float('Inf') or loss.data[0] is float('NaN'):
                print('EARLY STOP because of invalid loss value')
                valid_training = False
                break

            # update the GRADIENTS
            loss.backward()
            # update the WEIGHTS
            optimizer.step()

            # print progress
            running_loss += loss.data[0]
            if i % step == (step - 1):
                print('[%3d/%3d, %3d/%3d] loss: %.3f' % (epoch + 1, nb_epochs, i + 1, max_steps, running_loss / step))
                running_loss = 0.0

        # if we do not need to stop the training, compute the evaluation of the model
        # and save the history of the loss (both for training and testing)
        if valid_training:
            # Testing.
            print('Evaluating...')
            train_predictions = predict(net, train_set)
            train_loss, train_mae = evaluate(train_predictions, torch.FloatTensor(train_set.train_values))
            test_predictions = predict(net, test_set)
            test_loss, test_mae = evaluate(test_predictions, torch.FloatTensor(test_set.test_values))
            print('Loss =>\tTrain: %.3f\tTest: %.3f' % (train_loss, test_loss))
            print('MAE ==>\tTrain: %.2f\tTest: %.2f' % (train_mae, test_mae))
            loss_history = [[train_loss, test_loss]] if loss_history is None else loss_history + [[train_loss, test_loss]]

            # check if the network is still learning
            if test_loss < best_test_loss and train_loss < best_train_loss:
                print('Saving checkpoint at epoch', epoch)
                torch.save(net.state_dict(), 'checkpoint.ckpt')
                best_train_loss = train_loss
                best_test_loss = test_loss
                best_epoch_id = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # note that even if we prefer to stop the training, it is considered valid
            if epochs_without_improvement >= max_epochs_without_improvement:
                optimizer = lr_scheduler(optimizer, lr_decay=0.5)
                epochs_without_improvement = 0
                #print('EARLY STOP because the network does not learn anymore')
                #break

    print('Finished training')

    if valid_training:
        print('Saving final model...')
        torch.save(net.state_dict(), model_name)
        print('Best trained model at epoch', best_epoch_id, 'with test loss', best_test_loss)
        
        xx = np.linspace(0, len(loss_history) - 1, len(loss_history))
        loss_history = np.array(loss_history).astype(np.float32)

        plt.figure()
        plt.grid(True)
        plt.plot(xx, loss_history[:, 0], color='b', label='train')
        plt.plot(xx, loss_history[:, 1], color='r', label='test')
        plt.legend()
        plt.title('Training results')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


#%% ---------------------------------------------------------------------------
# Testing.


print('Evaluating testing set...')
test_loss, test_mae = evaluate(net, test_set)
print('Loss:', test_loss)
print('Mean Absolute Error:', test_mae)

preds = predict(network, test_set)
#errors = preds.data - test_set.test_values

info = 'info, test_loss, test_mae, preds, test_values'
torch.save((info, test_loss, test_mae, preds, test_set.test_values), 'results.pth')
