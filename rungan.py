# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:22:50 2019

@author: Manisha
"""

#%load_ext autoreload
from src.utils import VideoInterpTripletsDataset
from src.train import trainGAN
from src.eval import evalGAN

from torch.utils.data import DataLoader
import torch
#%autoreload 2

dataset = VideoInterpTripletsDataset('datasets/frames/train', read_frames=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
valset = VideoInterpTripletsDataset('datasets/frames/val',read_frames=True)
valloader = DataLoader(valset,batch_size=32,shuffle=True,num_workers=4) 

#print(dataloader)
#print(valloader)


generator, discriminator = trainGAN(25, dataloader, valloader=valloader,supervised=True, save_path='./experiments/v6', save_every=1, gan=True, unet=False)

#Runeval.py for eval
