# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:17:59 2020

@author: jplineb
"""

import torch
import torchvision.transforms as transforms
from PIL import *
import matplotlib.pyplot as plt
from sigmoid_1to5 import *
import imutils
import cv2

## define scale down to tensor object
scaledown2Tensor = transforms.Compose([
    transforms.Resize((72,128)),
    transforms.ToTensor()])

## define loading img onto gpu
def getreadyforpred (input_img):
  color_switch = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB) #switch the colors
  PIL_img = Image.fromarray(color_switch) # Converts array to PIL image
  pred_img = scaledown2Tensor(PIL_img)[None].cuda() #load as tensor on GPU
  return(pred_img)
  
  


