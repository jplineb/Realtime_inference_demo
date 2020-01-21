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

## do inference function
def livevideoinference(filepath, model_path):
  model = torch.load(model_path) #loads torch model
  model.eval() # sets model to evaluation
  all_predictions = [] # create empty array for predictions
  cap = cv2.VideoCapture(filepath)
  amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # get total frames from video
  amount_of_frames = 120 # TEST: Cap frames
  frame_number = 1 # sets starting frame
  while frame_number <= amount_of_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    _,frame = cap.read()
    frame = getreadyforpred(frame)
    inference = model(frame)
    all_predictions.append((frame_number,inference.item()))
    frame_number+=1
    print(frame_number)
  return(all_predictions)  

## graph function for output of tuples
    ## Graph predictions vs frames

def create_graph (aroftups):
    x_val = [x[0] for x in aroftups]
    y_val = [x[1] for x in aroftups]
    # labels
    plt.ylabel('Predicted Label')
    plt.xlabel('Frame Number')
    # plot
    plt.plot(x_val,y_val)
    plt.show()
    
    
if __name__ == "__main__":
    
