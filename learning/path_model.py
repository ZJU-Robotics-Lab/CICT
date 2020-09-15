#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)
        
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.LeakyReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.cbam = CBAM(planes, 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64,  layers[0], stride=5)
        self.layer2 = self._make_layer(128, layers[1], stride=5)
        self.layer3 = self._make_layer(256, layers[2], stride=3)
        self.layer4 = self._make_layer(256, layers[3], stride=3)

        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        #init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x

def ResidualNet():
    model = ResNet([2, 2, 2, 2])
    return model

class ModelGRU(nn.Module):
    def __init__(self, hidden_dim=256):
        super(ModelGRU, self).__init__()
        self.cnn_feature_dim = hidden_dim
        self.rnn_hidden_dim = hidden_dim
        self.cnn = CNN(input_dim=1, out_dim=self.cnn_feature_dim)
        #self.cnn = ResidualNet()
        self.gru = nn.GRU(
            input_size = self.cnn_feature_dim, 
            hidden_size = self.rnn_hidden_dim, 
            num_layers = 2,
            batch_first=True,
            dropout=0.2)
        self.mlp = MLP_COS(input_dim=self.rnn_hidden_dim+2)

    def forward(self, x, t, v0):
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        x = F.leaky_relu(x[:, -1, :])
        x = self.mlp(x, t, v0)
        return x
    
class ModelGMM(nn.Module):
    def __init__(self, k=10, hidden_dim=256):
        super(ModelGMM, self).__init__()
        self.k = k
        self.cnn_feature_dim = hidden_dim
        self.rnn_hidden_dim = hidden_dim
        self.cnn = CNN(input_dim=1, out_dim=self.cnn_feature_dim)
        #self.cnn = ResidualNet()
        self.gru = nn.GRU(
            input_size = self.cnn_feature_dim, 
            hidden_size = self.rnn_hidden_dim, 
            num_layers = 2,
            batch_first=True,
            dropout=0.2)
        self.mlp = MLP_GMM(input_dim=self.rnn_hidden_dim+2, k=self.k)

    def forward(self, x, t, v0):
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        x = F.leaky_relu(x[:, -1, :])
        x = self.mlp(x, t, v0)
        return x
  
class CNN_AVG(nn.Module):
    def __init__(self,input_dim=1, out_dim=256):
        super(CNN_AVG, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.avg_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = x.view(-1, self.out_dim)
        return x
    
class CNN(nn.Module):
    def __init__(self,input_dim=1, out_dim=256):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = x.view(-1, self.out_dim)
        return x
        
class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.linear1 = nn.Linear(256+1, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 511)
        
        self.linear4 = nn.Linear(512, 512)
        self.linear5 = nn.Linear(512, 512)
        self.linear6 = nn.Linear(512, 512)
        
        self.linear7 = nn.Linear(512, 2)
        
        self.apply(weights_init)
        
    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x, t], dim=1)
        
        x = self.linear4(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear6(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear7(x)
        
        #x = torch.tanh(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim=257):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 512)
        self.linear5 = nn.Linear(512, 1)
        self.linear6 = nn.Linear(512, 1)
        
        self.apply(weights_init)
        
    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        #x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        _x = self.linear5(x)
        _x = torch.sigmoid(_x)
        _y = self.linear6(x)
        _y = torch.tanh(_y)
        return torch.cat([_x, _y], dim=1)

class MLP_COS(nn.Module):
    def __init__(self, input_dim=257, rate=1.0):
        super(MLP_COS, self).__init__()
        self.rate = rate
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 4)
        
        self.apply(weights_init)
        
    def forward(self, x, t, v0):
        x = torch.cat([x, t], dim=1)
        x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(self.rate*x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        return x
    
class MLP_GMM(nn.Module):
    def __init__(self, input_dim=257, k=10, rate=1.0):
        super(MLP_GMM, self).__init__()
        self.k = k
        self.rate = rate
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 3*2*self.k)
        self.apply(weights_init)
        
    def forward(self, x, t, v0):
        x = torch.cat([x, t], dim=1)
        x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(self.rate*x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        
        weights_x = x[:,:self.k]
        weights_y = x[:,self.k:2*self.k]
        mu_x = x[:,2*self.k:3*self.k]
        mu_y = x[:,3*self.k:4*self.k]
        logvar_x = x[:,4*self.k:5*self.k]
        logvar_y = x[:,5*self.k:]
        return weights_x, weights_y, mu_x, mu_y, logvar_x, logvar_y
    
class Model_COS(nn.Module):
    def __init__(self,rate=1.0):
        super(Model_COS, self).__init__()
        #resnet = models.resnet34(pretrained=True)
        #self.cnn = nn.Sequential(*list(resnet.children())[0:9])
        self.cnn = CNN()
        self.mlp = MLP_COS(rate)
    
    def forward(self, x, t, v0):
        x = self.cnn(x)
        #x = x.view(-1, 512)
        x = self.mlp(x, t, v0)
        return x
    
class Model_COS_Img(nn.Module):
    def __init__(self,rate=1.0):
        super(Model_COS_Img, self).__init__()
        self.cnn = CNN(input_dim=6)
        self.mlp = MLP_COS(rate)
    
    def forward(self, x, t, v0):
        x = self.cnn(x)
        x = self.mlp(x, t, v0)
        return x
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN()
        self.mlp = MLP()
    
    def forward(self, x, t):
        x = self.cnn(x)
        x = self.mlp(x, t)
        return x
    
class Sine(nn.Module):
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class SIN_MLP(nn.Module):
    def __init__(self, input_dim=129):
        super(SIN_MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 128)
        self.linear5 = nn.Linear(128, 1)
        self.linear6 = nn.Linear(128, 1)
        self.activation = Sine()
        self.apply(weights_init)
        
    def forward(self, x):
        #x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        #x = self.activation(x)
        x = F.leaky_relu(x)
        _x = self.linear5(x)
        _x = torch.sigmoid(_x)
        _y = self.linear6(x)
        _y = torch.tanh(_y)
        #x = torch.tanh(x)
        return torch.cat([_x, _y], dim=1)
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.cnn = CNN()
        """
        self.mlp = nn.Sequential(
            nn.Linear(128+1, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )
        """
        self.mlp = SIN_MLP()
        self.fc_mu = nn.Linear(256, 128)
        self.fc_log_var = nn.Linear(256, 128)
        
    def encode(self, x):
        h = self.cnn(x)
        return self.fc_mu(h), self.fc_log_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x, t):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_t = torch.cat([z, t], dim=1)
        return self.mlp(z_t), mu, logvar
