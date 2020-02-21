#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    
    
    def __init__(self,in_size, H_size):
        """ 
        Apply the output of the convolution later (x_conv) through a highway network
                @param in_size (int): Size of input layer; it's e_{word} (dimensionality)
                @param H_size (int): Size of Hidden layer; it's e_{word} (dimensionality)
        """
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.proj = F.relu(nn.linear(in_size,H_size))
        self.gate = nn.Sigmoid(nn.linear(nn.linear(in_size,H_size)))
        
    def forward(self,X_conv_out):
        """ 
        Apply the output of the convolution later (X_conv_out) through a highway network
                @param X_conv_out (Tensor): Input X_conv_out gets applied to Highway network - shape of input tensor [batch_size,1,e_word]
                @returns X_word_emb (Tensor): Size of e_word, same as the size of hidden layer -- NOTE: check the shapes
        """
        
        x_proj = self.proj(X_conv_out)
        x_gate = self.gate(X_conv_out)
        x_highway = x_proj * x_gate + (1 - x_gate) * X_conv_out # a * b == torch.mul(a, b); 
        
        return x_highway # size [batch_size,1,e_word]
        
    ### END YOUR CODE

