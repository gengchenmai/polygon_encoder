import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

class Norm(nn.Module):
    def __init__(self, mean=0, std=1):
        super(Norm, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

class LeNet5(nn.Module):
    def __init__(self, in_channels, num_classes, signal_sizes=(28,28), hidden_dim = 250, mean=0, std=1):
        super(LeNet5, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.norm = Norm(mean, std)
        self.conv2_drop = nn.Dropout2d()
        
        self.conv_embed_dim = self.compute_conv_embed_dim(signal_sizes, curent_channels = 20)
        self.fc1 = nn.Linear(self.conv_embed_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        self.signal_sizes = signal_sizes
    
    def compute_conv_embed_dim(self, signal_sizes, curent_channels):
        fx, fy = signal_sizes
        return math.floor((fx-12)/4) * math.floor((fy-12)/4) * curent_channels

    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, in_channels, fx, fy), image tensor
                signal_sizes = (fx, fy)
        Return:
            x: shape (batch_size, num_classes)
                image class distribution

        '''
        batch_size, n_c, fx, fy = x.shape
        assert n_c == self.in_channels
        assert fx == self.signal_sizes[0]
        assert fy == self.signal_sizes[1]
        
        # x = x.view(-1, 1, self.signal_sizes[0], self.signal_sizes[1])

        # x: shape (batch_size, in_channels, fx, fy)
        x = self.norm(x)
        # self.conv1(x): shape [batch_size, 10, fx-4, fy-4]
        # F.max_pool2d(self.conv1(x), 2): shape [batch_size, 10, (fx-4)/2, (fy-4)/2 ]
        # x: shape [batch_size, 10, (fx-4)/2, (fy-4)/2 ]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # self.conv2(x): [batch_size, 20, (fx-12)/2, (fy-12)/2 ]
        # self.conv2_drop(self.conv2(x)): [batch_size, 20, (fx-12)/2, (fy-12)/2 ]
        # F.max_pool2d(self.conv2_drop(self.conv2(x)), 2): [batch_size, 20, (fx-12)/4, (fy-12)/4 ]
        # x: [batch_size, 20, (fx-12)/4, (fy-12)/4 ]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        
        _, n_cc, fx_, fy_ = x.shape
        assert n_cc == 20
        assert fx_ == math.floor((fx-12)/4)
        assert fy_ == math.floor((fy-12)/4)
        
        # x: shape (batch_size, conv_embed_dim = 20 * floor((fx-12)/4) * floor((fy-12)/4) )
        x = x.reshape(batch_size, -1)
        # x: shape (batch_size, 250)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # x: shape (batch_size, num_classes)
        x = self.fc2(x)
        return x