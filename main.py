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


#%% ---------------------------------------------------------------------------
# Load dataset.


nb_epochs = 100
batch_size = 32
shuffle_train_set = True
use_batch_norm = True
use_dropout = False
use_vgg16_basemodel = False

nb_channels = 3 if use_vgg16_basemodel else 1


# i dati sono già normalizzati tra 0 e 1, quindi rimuovo 0.5 per centrare in 0 l'intero dataset
# e moltiplico per 2 (dividendo per 0.5) in modo da scalare il dataset tra -1 e +1
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# scarico il dataset e lo trasformo in fase di caricamento
#train_set = dataset.FACES(train=True, transform=transf)
train_set = dataset.FATSYNTH('HeadLegArmLess', nb_channels, train=True, transform=transf)
# il dataset è salvato come una struttura su cui iterare, quindi creo un oggetto capace di leggerlo
# in maniera iterativa, impostando il numero di thread da lanciare per questo compito
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train_set, num_workers=0)

# come per il training set, ma carico il testing set
#test_set = dataset.FACES(train=False, transform=transf)
test_set = dataset.FATSYNTH('HeadLegArmLess', nb_channels, train=False, transform=transf)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


#%% ---------------------------------------------------------------------------
# Show some samples.


import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5 # unnomalize (riporto tra 0 e 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# estrai e visualizza alcune immagini campione
dataiter = iter(train_loader) # il primo oggetto è l'iteratore vero e proprio
images, targets = dataiter.next()

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


# ogni rete è un "modulo"
class Net(nn.Module):
    
    def __init__(self, input_shape, vgg16_basemodel=True, batch_normalization=False, dropout=False):
        # definisco le componenti della rete (dimensionalità)
        # ma non specifico ancora come e dove dovranno essere
        # calcolati
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
            self.pool = nn.MaxPool2d(2, 2) # handle per il pooling
            self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
            self.conv1_bn = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.conv2_bn = nn.BatchNorm2d(64)

        # calcolo una volta il risultato del forward pass relativo
        # alle sole features della rete, così da stimare il numero
        # delle features che andranno processate dal classificatore
        x = self._features(Variable(torch.zeros(1, *input_shape)))
        self.nfts = x.numel()
        
        self.fc1 = nn.Linear(self.nfts, 1024)
        self.fc2 = nn.Linear(1024, 256)
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
    
    # definisco il comportamento della rete, potendo dividere anche il
    # processing in base all'hardware disponibile (GPU)
    def forward(self, x):
        x = self._features(x)
        # "view" è una funzione cost-free (come "reshape" per numpy)
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


target_shape = (nb_channels,) + (224, 224)
net = Net(input_shape=target_shape, vgg16_basemodel=use_vgg16_basemodel, batch_normalization=use_batch_norm, dropout=use_dropout)
net.apply(weights_init) # applica per ogni modulo la funzione passata come parametro
print(net)


if HAS_CUDA:
    # tutti i tensori della rete sono convertiti automaticamente
    # in tensori cuda, adatti al calcolo su GPU
    net.cuda(gpu_id)


#%% ---------------------------------------------------------------------------
# Training.


import torch.optim as optim


class BiweightLoss(nn.Module):
    """Biweight loss function.
     Based on: https://arxiv.org/abs/1505.06606
     """
    def __init__(self, C=4.6851):
        super(BiweightLoss, self).__init__()
        self.C = C if C is float else None

    def forward(self, input, target):
        # compute the absolute residuals
        r = torch.abs(input - target)

        def median(x):
            x, _ = x.sort()
            mad_id = x.size() // 2
            return x[mad_id]

        # if C is not fixed, compute it as MAD of residuals
        if self.C is None:
            mad = median(torch.abs(r - median(r)))
            self.C = mad # residual mad
        
        # useful function to assign the right value to a tensor
        # according to a specified condition "cond"
        def where(cond, x1, x2):
            return (cond.float() * x1) + ((1 - cond.float()) * x2)

        residuals = where(r < 1,
            0.5 * torch.pow(r, 2),
            self.C * (r - 0.5 * self.C))

        loss = torch.mean(residuals)
        return loss


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


def evaluate(network, dataset, batch_size=8):
    cum_loss = 0
    cum_absolute_error = 0
    count = 0
    network.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for data in loader:
        images, targets = data
        targets = Variable(targets)
        if HAS_CUDA:
            images, targets = images.cuda(gpu_id), targets.cuda(gpu_id)
        outputs = network(Variable(images))
        cum_loss += criterion(outputs, targets).data[0]
        cum_absolute_error += torch.sum(torch.abs(outputs.data - targets.data))
        count += 1

    loss, mean_absolute_error = (cum_loss / count), (cum_absolute_error / count)

    return loss, mean_absolute_error


model_name = 'trained_model.pth'
if os.path.exists(model_name):

    print('Found pretrained model. Loading file', model_name)
    net.load_state_dict(torch.load(model_name))

else:

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
            
            # for each batch, update the learning rate
            optimizer = lr_scheduler(optimizer, lr_decay=0.001)

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
                print('[%2d/%2d, %4d/%4d] loss: %.3f' % (epoch + 1, nb_epochs, i + 1, max_steps, running_loss / step))
                running_loss = 0.0

        # if we do not need to stop the training, compute the evaluation of the model
        # and save the history of the loss (both for training and testing)
        if valid_training:
            # Testing.
            print('Evaluating...')
            train_loss, train_mae = evaluate(net, train_set)
            test_loss, test_mae = evaluate(net, test_set)
            print('Loss =>\tTrain: %.3f\tTest: %.3f' % (train_loss, test_loss))
            print('MAE ==>\tTrain: %.1f\tTest: %.1f' % (train_mae, test_mae))
            loss_history = [[train_loss, test_loss]] if loss_history is None else loss_history + [[train_loss, test_loss]]

            if test_loss < best_test_loss:
                print('Saving checkpoint at epoch', epoch)
                torch.save(net.state_dict(), 'checkpoint.pth')
                best_test_loss = test_loss
                best_epoch_id = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # note that even if we prefer to stop the training, it is considered valid
            if epochs_without_improvement >= max_epochs_without_improvement:
                print('EARLY STOP because the network does not learn anymore')
                break

    print('Finished training')

    if valid_training:
        print('Saving final model...')
        torch.save(net.state_dict(), model_name)
        print('Best trained model at epoch', best_epoch_id, 'with test loss', best_test_loss)

        import matplotlib.pyplot as plt
        import numpy as np

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
