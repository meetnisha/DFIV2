# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:22:50 2019

@author: Monu
"""

#%load_ext autoreload
from src.utils import VideoInterpTripletsDataset
from src.train import trainGAN
from src.eval import evalGAN

from torch.utils.data import DataLoader
import torch
import os
data_dir = '../'

valset = VideoInterpTripletsDataset('datasets/frames/val',read_frames=True)
valloader = DataLoader(valset,batch_size=32,shuffle=True,num_workers=4)

print(evalGAN(valloader,load_path = './models/Model_SGAN/20_Generator',sampleImagesName="SGAN_val", unet=True))

testset = VideoInterpTripletsDataset('datasets/frames/test',read_frames=True)
testloader  = DataLoader(testset,batch_size=32,shuffle=True,num_workers=4)

evalGAN(testloader, './models/Model_SGAN/20_Generator',sampleImagesName="SGAN_test", unet=True)