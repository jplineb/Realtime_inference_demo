# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:17:59 2020

@author: jplineb
"""

import torch
import torchvision.transforms as transforms
from PIL import *
import matplotlib.pyplot as plt

import sigmoid_1to5
import imutils
import cv2
import os
import math

## Get list of available video files
def getfiles(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        print(root)
        print(dirs)
        for file in files:
            print(file)
            if file.endswith('.mp4'):
                file_list.append(os.path.join(root,file))
    return(file_list)


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
  # Single Video
def livevideoinference(filepath, model):
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
  
  # Multiple videos
def databasevideoinfernce(filepaths, model):
  all_inf = []
  for x in filepaths:
      all_predictions = []
      print(x[0])
      quality = x.split('/')[6]
      cap = cv2.VideoCapture(x)
      amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      frame_number = 1
      while frame_number <= amount_of_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        _,frame = cap.read()
        frame = getreadyforpred(frame)
        inference = model(frame)
        all_predictions.append([quality,(frame_number,inference.item())]) #creates and stores all predictions in a matrix
        frame_number+=1
        print(frame_number)
      all_inf.append(all_predictions) # Stores video in its own dataset inside matrix
    
  return(all_inf)


### Graph predictions vs frames
  ## Single use case
def create_graph (aroftups):
    x_val = [x[0] for x in aroftups]
    y_val = [x[1] for x in aroftups]
    # labels
    plt.ylabel('Predicted Label')
    plt.xlabel('Frame Number')
    # plot
    plt.plot(x_val,y_val)
    plt.show()
    
  ## Graph subplots of mutiple predictions
def create_suplots(all_inferences):
    fig = plt.figure(figsize=(40,40))
    am_of_plts = len(all_inf)
    plot_dim = math.ceil(math.sqrt(am_of_plts))
    plot = 1
    for alpha in all_inferences:
        # point towards sublplot
        plt.subplot(plot_dim, plot_dim, plot)
        # get values for plot
        x_val = [x[1][0] for x in alpha]
        y_val = [x[1][1] for x in alpha]
        # labels
        plt.title(alpha[0][0])
        plt.ylabel('Predicted Label')
        plt.xlabel('Frame Number')
        plt.ylim(0,5)
          # plot
        plt.plot(x_val,y_val)
        plot+=1
    plt.show()
        
        
    
if __name__ == "__main__":
    video_path = './test_video.mp4'
    all_inf=[]
    model = torch.load('/content/drive/My Drive/FASTAI /Realtime_demo/test_model_v2.pt')
    model.eval()
    model_preds = livevideoinference(video_path, model)
    create_graph(model_preds)
    
