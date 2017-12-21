import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# to build a network, extend the class nn.Module
class Net(nn.Module):
    
    def __init__(self, input_shape, vgg16_basemodel=True, batch_normalization=False, dropout=False):
        # define the network components but not the actual architecture
        super(Net, self).__init__()

        self.use_base_model = vgg16_basemodel
        self.use_batch_normalization = batch_normalization
        self.use_dropout = dropout
        
        self.base_model = None
        if self.use_base_model:
            if self.use_batch_normalization:
                self.base_model = models.vgg16_bn(pretrained=True).features
            else:
                self.base_model = models.vgg16(pretrained=True).features
            self.enable_base_model_training(True)
        else:
            self.pool = nn.MaxPool2d(2, 2)
            
            self.conv3x3_1 = nn.Conv2d(input_shape[0], 8, 3, padding=1)
            self.conv3x3_1_bn = nn.BatchNorm2d(8)
            self.conv5x5_1 = nn.Conv2d(input_shape[0], 8, 5, padding=2)
            self.conv5x5_1_bn = nn.BatchNorm2d(8)
            self.conv7x7_1 = nn.Conv2d(input_shape[0], 8, 7, padding=3)
            self.conv7x7_1_bn = nn.BatchNorm2d(8)
            
            self.conv3x3_2 = nn.Conv2d(8, 16, 3, padding=1)
            self.conv3x3_2_bn = nn.BatchNorm2d(16)
            self.conv5x5_2 = nn.Conv2d(8, 16, 5, padding=2)
            self.conv5x5_2_bn = nn.BatchNorm2d(16)
            self.conv7x7_2 = nn.Conv2d(8, 16, 7, padding=3)
            self.conv7x7_2_bn = nn.BatchNorm2d(16)

        # to compute the number of vectorized features we need to compute
        # once the forward pass on the feature_extractor
        x = self._features(torch.autograd.Variable(torch.zeros(1, *input_shape)))
        self.nfts = x.numel()
        
        self.fc1 = nn.Linear(self.nfts, 256)
        self.fc1_bn = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm2d(128)
        self.fc3 = nn.Linear(128, 2)
        
        # initialize weights
        self.reset_weights()
    
    def _features(self, x):
        if self.use_base_model:
            x = self.base_model(x)
        else:
            if self.use_batch_normalization:
                feat_3x3 = self.pool(F.relu(self.conv3x3_1_bn(self.conv3x3_1(x))))
                feat_5x5 = self.pool(F.relu(self.conv5x5_1_bn(self.conv5x5_1(x))))
                feat_7x7 = self.pool(F.relu(self.conv7x7_1_bn(self.conv7x7_1(x))))
            else:
                feat_3x3 = self.pool(F.relu(self.conv3x3_1(x)))
                feat_5x5 = self.pool(F.relu(self.conv5x5_1(x)))
                feat_7x7 = self.pool(F.relu(self.conv7x7_1(x)))

            x = (feat_3x3 + feat_5x5 + feat_7x7) / 3.

            if self.use_batch_normalization:
                feat_3x3 = self.pool(F.relu(self.conv3x3_2_bn(self.conv3x3_2(x))))
                feat_5x5 = self.pool(F.relu(self.conv5x5_2_bn(self.conv5x5_2(x))))
                feat_7x7 = self.pool(F.relu(self.conv7x7_2_bn(self.conv7x7_2(x))))
            else:
                feat_3x3 = self.pool(F.relu(self.conv3x3_2(x)))
                feat_5x5 = self.pool(F.relu(self.conv5x5_2(x)))
                feat_7x7 = self.pool(F.relu(self.conv7x7_2(x)))

            x = (feat_3x3 + feat_5x5 + feat_7x7) / 3.
            
        return x
    
    def _regressor(self, x):

        if self.use_batch_normalization:
            x = F.relu(self.fc1_bn(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x)

        if self.use_batch_normalization:
            x = F.relu(self.fc2_bn(self.fc2(x)))
        else:
            x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = F.dropout(x)

        x = self.fc3(x)
        return x
    
    # compute at runtime the forward pass
    def forward(self, x):
        x = self._features(x)
        # "view" is a cost-free function (as "reshape" in numpy)
        x = x.view(-1, self.nfts)
        x = self._regressor(x)
        return x

    # eanble/disable the training of the base_model (if any)
    def enable_base_model_training(self, enable=True):
        if self.base_model is not None:
            for param in self.base_model.parameters():
                self.base_model.requires_grad = enable
    
    def _normal_init(self, module, mu=0., std=0.01):
        module.weight.data.normal_(mu, std)
        module.bias.data.fill_(mu)
    
    def _xavier_init(self, module):
        if len(module.weight.data.shape) > 1:
            N_in = module.weight.data.size()[1]
            N_out = module.weight.data.size()[0]
            N = (N_in + N_out) / 2
        else:
            N = module.weight.data.size()[0]
        xavier_var = 1. / N
        xavier_std = math.sqrt(xavier_var)
        module.weight.data.normal_(0.0, xavier_std)
        module.bias.data.fill_(0.0)
    
    def reset_weights(self):
        
        if not self.use_base_model:
            self._normal_init(self.conv3x3_1)
            self._normal_init(self.conv3x3_1_bn)
            self._normal_init(self.conv5x5_1)
            self._normal_init(self.conv5x5_1_bn)
            self._normal_init(self.conv7x7_1)
            self._normal_init(self.conv7x7_1_bn)

            self._normal_init(self.conv3x3_2)
            self._normal_init(self.conv3x3_2_bn)
            self._normal_init(self.conv5x5_2)
            self._normal_init(self.conv5x5_2_bn)
            self._normal_init(self.conv7x7_2)
            self._normal_init(self.conv7x7_2_bn)
        
        self._normal_init(self.fc1)
        self._normal_init(self.fc1_bn)
        
        self._normal_init(self.fc2, 0.0, 0.001)


class INet(torch.nn.Module):

    def __init__(self, input_shape, nb_classes):
        super(INet, self).__init__()

        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.nfts = -1
        self.features = None
        self.classifier = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.nfts)
        x = self.classifier(x)
        return x

    def _normal_init(self, module, mu=0.0, std=0.01):
        class_name =  module.__class__.__name__
        for module_name in ['ReLU', 'MaxPool', 'Sequential']:
            if class_name.find(module_name) != -1:
                return
        module.weight.data.normal_(mu, std)
        module.bias.data.fill_(mu)
    
    def _xavier_init(self, module):
        class_name =  module.__class__.__name__
        for module_name in ['ReLU', 'MaxPool', 'Sequential']:
            if class_name.find(module_name) != -1:
                return
        import math
        if len(module.weight.data.shape) > 1:
            N_in = module.weight.data.size()[1]
            N_out = module.weight.data.size()[0]
            N = (N_in + N_out) / 2
        else:
            N = module.weight.data.size()[0]
        xavier_var = 1. / N
        xavier_std = math.sqrt(xavier_var)
        module.weight.data.normal_(0.0, xavier_std)
        module.bias.data.fill_(0.0)


class MobileNet(INet):

    def __init__(self, input_shape, nb_classes):
        INet.__init__(self, input_shape, nb_classes)

        def conv_bn(ch_inp, ch_out, stride):
            return torch.nn.Sequential(
                torch.nn.Conv2d(ch_inp, ch_out, 3, stride, 1, bias=False),
                torch.nn.BatchNorm2d(ch_out),
                torch.nn.ReLU(inplace=True)
            )

        def conv_dw(ch_inp, ch_out, stride):
            return torch.nn.Sequential(
                torch.nn.Conv2d(ch_inp, ch_inp, 3, stride, 1, groups=ch_inp, bias=False),
                torch.nn.BatchNorm2d(ch_inp),
                torch.nn.ReLU(inplace=True),
    
                torch.nn.Conv2d(ch_inp, ch_out, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(ch_out),
                torch.nn.ReLU(inplace=True),
            )

        ch = self.input_shape[0]

        self.features = torch.nn.Sequential(
            conv_bn(  ch,   32, 2), 
            conv_dw(  32,   64, 1),
            conv_dw(  64,  128, 2),
            conv_dw( 128,  128, 1),
            conv_dw( 128,  256, 2),
            conv_dw( 256,  256, 1),
            conv_dw( 256,  512, 2),
            conv_dw( 512,  512, 1),
            conv_dw( 512,  512, 1),
            conv_dw( 512,  512, 1),
            conv_dw( 512,  512, 1),
            conv_dw( 512,  512, 1),
            conv_dw( 512, 1024, 2),
            conv_dw(1024, 1024, 1),
            torch.nn.AvgPool2d(4))

        x = self.features(torch.autograd.Variable(torch.zeros(1, *self.input_shape)))
        self.nfts = x.numel()

        self.classifier = torch.nn.Linear(self.nfts, self.nb_classes)

        self.fc1 = nn.Linear(self.nfts, int(self.nfts / 4))
        self.fc1_bn = nn.BatchNorm2d(int(self.nfts / 4))
        self.fc2 = nn.Linear(int(self.nfts / 4), 128)
        self.fc2_bn = nn.BatchNorm2d(128)
        self.fc3 = nn.Linear(128, 128)
        self.fc3_bn = nn.BatchNorm2d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc4_bn = nn.BatchNorm2d(64)
        self.fc5 = nn.Linear(64, 1)

        self.classifier = torch.nn.Sequential(
            self.fc1, self.fc1_bn, torch.nn.ReLU(inplace=True),
            self.fc2, self.fc2_bn, torch.nn.ReLU(inplace=True),
            self.fc3, self.fc3_bn, torch.nn.ReLU(inplace=True),
            self.fc4, self.fc4_bn, torch.nn.ReLU(inplace=True),
            self.fc5
            )

        self.classifier.apply(self._normal_init)
