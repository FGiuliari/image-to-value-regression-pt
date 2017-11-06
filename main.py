#!/usr/bin/python3


#%% ---------------------------------------------------------------------------
# Imports.


from __future__ import print_function

import torch
import torchvision
from torchvision import transforms

import dataset
import os

import numpy as np

#%% -------------------------- >>> MODIFY HERE <<< ----------------------------
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

# what are the tasks in this demo?
task = 'fat-from-depth' # age-from-faces, gender-from-depth, fat-from-depth

nb_epochs = 60 # max number of training epochs
batch_size = 16 # <== reduce this value if you encounter memory errors
shuffle_train_set = True
use_batch_norm = True
use_dropout = False
use_vgg16_basemodel = False
use_data_augmentation_hflip = True # WARNING - data augmentation doubles the batch size
use_early_stop_triggers = True

# name of the saved files
model_filename = 'network_state_dict.ckpt'
results_filename = 'results.pth'


# -------------------------------------
# Load dataset.


print('Loading data...')

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


# -------------------------------------
# Show some samples.


import matplotlib.pyplot as plt


'''
def imshow(img):
    img = img / 2 + 0.5 # unnomalize
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0))) # channel last


# extract some sample images
dataiter = iter(train_loader) # first item is the iterator
images, targets = dataiter.next() # extract batch

plt.figure()
imshow(torchvision.utils.make_grid(images))
#print(' '.join('%.3f' % targets[j] for j in range(batch_size)))


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
'''


#%% ---------------------------------------------------------------------------
# Setup network.


from torch.autograd import Variable
from network import Net # definition of the (custom) network architecture


print('Setting up network...')


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
net.apply(weights_init) # apply to each modules/elements the input function
net.enable_base_model_training(True)
print(net)


if HAS_CUDA:
    # all the tensors in the module are converted to cuda
    net.cuda(gpu_id)


#%% ---------------------------------------------------------------------------
# Training.


criterion = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.0005)


def lr_scheduler(optimizer, lr_decay=0.001, epoch=None, step=1):
    """Decay learning rate by a factor of lr_decay every step epochs.
    """
    if epoch is None or step == 1 or (epoch+1) % step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (1 - lr_decay)
    return optimizer


def predict(network, dataset, batch_size=1):
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
    '''Compute the mean absolute error of the predictions.
    The parameters must be pytorch tensors.
    '''
    loss = criterion(Variable(predictions, requires_grad=False), Variable(ground_truth_values, requires_grad=False))
    residuals = predictions - ground_truth_values
    abs_error = torch.abs(residuals)
    return loss.data[0], torch.mean(abs_error)


# verify that the model exists
if os.path.exists(model_filename):

    print('Found pretrained model. Loading file', model_filename)
    net.load_state_dict(torch.load(model_filename))

else:

    print('Start training')

    best_train_loss = float('Inf')
    best_test_loss = float('Inf')
    best_epoch_id = 0

    loss_history = None

    # use these variables as triggers to make a decision on the training
    epochs_without_improvement = 0
    max_epochs_without_improvement = 5
    valid_training = True

    # training loop
    for epoch in range(nb_epochs):

        # early stop check
        if not valid_training:
            break

        print('\nEPOCH', epoch + 1)

        # at each epoch, update the optimizer learning rate
        optimizer = lr_scheduler(optimizer, lr_decay=0.25)

        # enable training mode (the function eval() freezes the weight gradients)
        net.train()

        # variables used to print the training progress
        running_loss = 0.0
        max_steps = len(train_loader)
        step = int(max_steps / 5) # printing interval

        # batch loop
        for i, data in enumerate(train_loader):
            inputs, targets = data
            
            # data augmentation: flip horizontally and concatenate the
            # results to the given input batch
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

            # check if the computation is going well
            if abs(loss.data[0]) == float('Inf') or loss.data[0] is float('NaN'):
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
                print('[%3d/%3d, %4d/%4d] loss: %.3f' % (epoch + 1, nb_epochs, i + 1, max_steps, running_loss / step))
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

            # check if the network is still learning: both the training and testing
            # losses should increase, otherwise update the trigger variable
            if test_loss < best_test_loss and train_loss < best_train_loss:
                print('Saving checkpoint at epoch', epoch)
                torch.save(net.state_dict(), model_filename) # best solution so far
                best_train_loss = train_loss
                best_test_loss = test_loss
                best_epoch_id = epoch
                epochs_without_improvement = 0
            else:
                # when the network struggle to learn, try to help the training
                # by reducing the learning rate
                optimizer = lr_scheduler(optimizer, lr_decay=0.1)
                epochs_without_improvement += 1

            # note that even if we prefer to stop the training, it is considered valid
            if epochs_without_improvement >= max_epochs_without_improvement:
                if use_early_stop_triggers:
                    print('EARLY STOP because the network does not learn anymore')
                    break
                else:
                    # instead of stop the training, update the learning rate to try
                    # to improve the learning convergence
                    optimizer = lr_scheduler(optimizer, lr_decay=0.5)
                    epochs_without_improvement = 0

    print('\nFinished training')

    if valid_training:
        #print('Saving final model...')
        #torch.save(net.state_dict(), model_filename)
        print('Best trained model at [epoch %d] with test [loss %.3f]' % (best_epoch_id, best_test_loss))
        loss_history = np.array(loss_history).astype(np.float32)


#%% ---------------------------------------------------------------------------
# Testing.


if not os.path.exists(results_filename):

    print('\nEvaluating testing set...')
    test_predictions = predict(net, test_set)
    ground_truth_values = torch.FloatTensor(test_set.test_values)
    test_loss, test_mae = evaluate(test_predictions, ground_truth_values)

    info = 'info, test_loss, test_mae, preds, gt_values'
    torch.save((info, loss_history, test_loss, test_mae, test_predictions, ground_truth_values), 'results.pth')

else:

    data = torch.load(results_filename)
    info, loss_history, test_loss, test_mae, test_predictions, ground_truth_values = data

print('\nFinal results:')
print('Loss: %.3f' % test_loss)
print('MAE: %.2f' % test_mae)


#%% ---------------------------------------------------------------------------
# Visualization.


# show loss history
plt.figure()
plt.title('Loss history')
plt.xlabel('Epochs')
plt.ylabel('Smooth L1 Loss')
xx = np.linspace(0, loss_history.shape[0] - 1, loss_history.shape[0])
plt.grid(True)
plt.plot(xx, loss_history[:, 0], color='b', label='train')
plt.plot(xx, loss_history[:, 1], color='r', label='test')
plt.legend()

# show prediction results
plt.figure()
xx = np.linspace(1, test_predictions.numel(), test_predictions.numel())
gt = ground_truth_values.squeeze().numpy()
yy = test_predictions.squeeze().numpy()
idx = np.argsort(gt)
gt = gt[idx]
yy = yy[idx]
err = yy - gt
plt.subplot(2,1,1)
plt.title('Predictions')
plt.grid(True)
plt.plot(xx, gt, color='g', label='gt')
plt.plot(xx, yy, color='b', label='yy')
plt.legend()
plt.subplot(2,1,2)
plt.grid(True)
plt.plot(xx, err, color='r', label='err')
plt.legend()


# show bland-altman plot of predictions
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean  = np.mean([data1, data2], axis=0)
    diff  = data1 - data2        # Difference between data1 and data2
    md    = np.mean(diff)        # Mean of the difference
    sd    = np.std(diff, axis=0) # Standard deviation of the difference

    plt.figure()
    plt.title('Bland-Altman plot')
    plt.xlabel('mean')
    plt.ylabel('diff')
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')


bland_altman_plot(gt, yy)

plt.show()
