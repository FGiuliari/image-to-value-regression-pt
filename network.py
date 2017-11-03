import torch
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
        x = self._features(torch.autograd.Variable(torch.zeros(1, *input_shape)))
        self.nfts = x.numel()
        
        self.fc1 = nn.Linear(self.nfts, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)
    
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
        x = F.relu(self.fc4(x))
        if self.use_dropout:
            x = F.dropout(x)
        x = self.fc5(x)
        return x
    
    # compute at runtime the forward pass
    def forward(self, x):
        x = self._features(x)
        # "view" is a cost-free function (as "reshape" in numpy)
        x = x.view(-1, self.nfts)
        x = self._regressor(x)
        return x
