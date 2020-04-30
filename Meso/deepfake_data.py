import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset
import imageio
import numpy as np
import os


class deepfakeDataset(Dataset):

    def __init__(self, video_dir, split):
        
        if split =='train':
            #To Do add more transforms
            self.transforms = T.Compose([T.resize((256,256)),
                                        T.ToTensor()])
        elif split=='test':
            self.transforms = T.Compose([T.resize((256,256)),
                                        T.ToTensor()])




    def __getitem__(self):
        '''returns data which is a dictionary that contains the image and the ground truth
        if image is fake or real'''
        data = {}



        return data
