#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from siren_pytorch import SirenNet

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

class CNN(nn.Module):
    def __init__(self,input_dim=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        
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
        x = x.view(-1, 256)
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
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(256+1, 512)
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
    def __init__(self, rate=1.0):
        super(MLP_COS, self).__init__()
        self.rate = rate
        self.linear1 = nn.Linear(256+2, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 2)
        
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
    
class Model_COS(nn.Module):
    def __init__(self,rate=1.0):
        super(Model_COS, self).__init__()
        self.cnn = CNN()
        self.mlp = MLP_COS(rate)
    
    def forward(self, x, t, v0):
        x = self.cnn(x)
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
  
class CNN_SIN(nn.Module):
    def __init__(self):
        super(CNN_SIN, self).__init__()
        self.cnn = CNN()
        self.siren = SirenNet(
            dim_in = 256+1,                        # input dimension, ex. 2d coor
            dim_hidden = 256,                  # hidden dimension
            dim_out = 2,                       # output dimension, ex. rgb value
            num_layers = 5,                    # number of layers
            final_activation = nn.Tanh(),      # activation of final layer (nn.Identity() for direct output)
            w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        #self.mlp = SIN_MLP(input_dim=256+1)
    
    def forward(self, x, t):
        x = self.cnn(x)
        h = torch.cat([x, t], dim=1)
        x = self.siren(h)
        return x
