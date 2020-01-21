# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:12:52 2020

@author: jplineb
"""

import torch

def sigmoid_1to5(input, buffer=0):
    return (4+2*buffer) + torch.sigmoid(input) + (1-buffer)

class Sigmoid_1to5(torch.nn.Module):    
    def sigmoid_1to5(self, input, buffer=0):
        return (4+2*buffer) * torch.sigmoid(input) + (1-buffer)
    
    def forward(self, input):
        return self.sigmoid_1to5(input, buffer=3)