#%% Imports.


from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import dataset


#%% Load dataset.


nb_epochs = 10
batch_size = 1
shuffle_train_set = True
use_batch_norm = True
use_dropout = False
use_vgg16_basemodel = True


# i dati sono già normalizzati tra 0 e 1, quindi rimuovo 0.5 per centrare in 0 l'intero dataset
# e moltiplico per 2 (dividendo per 0.5) in modo da scalare il dataset tra -1 e +1
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# scarico il dataset e lo trasformo in fase di caricamento
train_set = dataset.FACES(train=True, transform=transf)
# il dataset è salvato come una struttura su cui iterare, quindi creo un oggetto capace di leggerlo
# in maniera iterativa, impostando il numero di thread da lanciare per questo compito
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train_set, num_workers=0)

# come per il training set, ma carico il testing set
test_set = dataset.FACES(train=False, transform=transf)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


#%% Show some samples.


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
print(' '.join('%.1f' % targets[j] for j in range(batch_size)))

Y = train_set.train_values
print(Y.shape, np.mean(Y), np.var(Y))
plt.figure()
plt.title('Age distribution')
plt.hist(Y, bins='auto')
plt.show()


#%% Setup network.


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
            self.conv1 = nn.Conv2d(input_shape[0], 64, 5)
            self.conv1_bn = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3)
            self.conv2_bn = nn.BatchNorm2d(128)

        # calcolo una volta il risultato del forward pass relativo
        # alle sole features della rete, così da stimare il numero
        # delle features che andranno processate dal classificatore
        x = self._features(Variable(torch.zeros(1, *input_shape)))
        self.nfts = x.numel()
        
        self.fc1 = nn.Linear(self.nfts, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
    
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


target_shape = (3, 181, 121)
net = Net(input_shape=target_shape, vgg16_basemodel=use_vgg16_basemodel, batch_normalization=use_batch_norm, dropout=use_dropout)
net.apply(weights_init) # applica per ogni modulo la funzione passata come parametro
print(net)


#%% Training.


import torch.optim as optim

criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
#optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)


def lr_scheduler(optimizer, epoch=None, lr_decay=0.001, step=1):
    """Decay learning rate by a factor of lr_decay every step epochs.
    If epoch is None, the learning rate is update immediately.
    """
    if epoch is None or (epoch+1) % step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (1 - lr_decay)
    return optimizer


HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

if HAS_CUDA:
    # tutti i tensori della rete sono convertiti automaticamente
    # in tensori cuda, adatti al calcolo su GPU
    net.cuda()


import os


def evaluate(network, dataset, batch_size=1):
    cum_loss = 0
    cum_absolute_error = 0
    count = 0
    network.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for data in loader:
        images, targets = data
        targets = Variable(targets)
        if HAS_CUDA:
            images, targets = images.cuda(), targets.cuda()
        outputs = network(Variable(images))
        cum_loss += criterion(outputs, targets).data[0]
        cum_absolute_error += torch.sum(torch.abs(outputs.data - targets.data))
        count += 1

    loss, mean_absolute_error = (cum_loss / count), (cum_absolute_error / count)

    return loss, mean_absolute_error


model_name = 'trained_model.pth'
if os.path.exists(model_name):

    net.load_state_dict(torch.load(model_name))

else:

    for epoch in range(nb_epochs):

        #optimizer = lr_scheduler(optimizer, epoch, lr_decay=0.5, step=2)

        net.train() # abilita il training del modello; eval() congela il training

        running_loss = 0.0
        max_steps = len(train_loader)
        step = int(max_steps / 5)
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            
            optimizer = lr_scheduler(optimizer, None, lr_decay=0.0005)

            if HAS_CUDA:
            # siccome la rete è inviata alla GPU, tutti i tensori che
            # utilizzerà dovranno a loro volta essere inviati alla scheda video
                inputs, targets = inputs.cuda(), targets.cuda()

            # il gradiente viene accumulato come comportamento di default, quindi
            # è necessario resettarlo ad ogni iterazione in modo che ogni batch di
            # campioni sia calcolata indipendentemente
            optimizer.zero_grad()

            # calcolo il forward pass della rete (output) convertendo l'input
            # in Variable, un tensore adatto al calcolo del gradiente
            outputs = net(Variable(inputs))
            # calcolo la loss sulla base dell'output e del ground truth
            loss = criterion(outputs, Variable(targets))
            # propago l'errore nella rete, aggiornando i GRADIENTI della rete
            loss.backward()
            # aggiorno i PESI della rete sulla base dei gradienti
            optimizer.step()

            # stampo lo stato di avanzamento, calcolando la loss media
            # su un certo numero di iterazioni
            running_loss += loss.data[0]
            if i % step == (step - 1):
                print('[%2d/%2d, %4d/%4d] loss: %.3f' % (epoch + 1, nb_epochs, i + 1, max_steps, running_loss / step))
                running_loss = 0.0

        # Testing.
        print('Evaluating...')
        train_loss, train_mae = evaluate(net, train_set, 8)
        test_loss, test_mae = evaluate(net, test_set, 8)
        print('Loss =>\tTrain: %.3f\tTest: %.3f' % (train_loss, test_loss))
        #print('MAE ==>\tTrain: %.1f\tTest: %.1f' % (train_mae, test_mae))

        print('Saving checkpoint...')
        torch.save(net.state_dict(), 'ckpt_' + str(epoch) + '.pth')

    print('Finished training')

    print('Saving final model...')
    torch.save(net.state_dict(), model_name)

print('DONE')
