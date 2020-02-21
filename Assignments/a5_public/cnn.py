#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        """
        Apply the output of the padded embedding lookup (x_emb) through a CNN network
        @param in_channels (int): number of in channels -- e_char
        @param out_channels (int): number of out channels (number of filters) -- e_word
        """
        self.conv1d = nn.Conv1d(in_channels,out_channels,kernel_size)
        self.maxpool = nn.MaxPool1d()
    def forward(self, x_reshaped):
        """
        Apply the output of the padded embedding lookup (x_reshaped) through a CNN network
        @param x_reshaped (Tensor): Input x_reshaped gets applied to CNN network - shape of input tensor [batch_size,e_char,m_word]
        @returns x_conv_out (Tensor): Size of (batch_size, e_word), the input for highway network
        """
        x_conv = self.conv1d(x_reshaped) # [batch_size, e_word, m_word-k+1]
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0] # [batch_size, e_word]
        return x_conv_out
    ### END YOUR CODE

