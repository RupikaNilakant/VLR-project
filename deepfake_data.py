import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms as T

from PIL import Image
from torch.utils.data import Dataset
import imageio
import numpy as np
import os


class deepfakeDataset(Dataset):

    def __init__(self, split, image_dir=None):

        #Note: this is based on the Meso datasets provided
        #may need to modify for other datasets
        if image_dir==None:
            self.base_dir = '/home/aubs/Downloads/deepfake_database/deepfake_database'
        else:
            self.base_dir = image_dir

        if split =='train':
            #To Do add more transforms
            #pdb.set_trace()
            self.transforms = T.Compose([T.Resize((256,256)),
                                        T.ToTensor()])
            self.image_dir = os.path.join(self.base_dir,'train:test')

            #list that holds full path to every image
            self.image_list = []
            #holds if real or fake
            self.gt_list = []

            df_path = os.path.join(self.image_dir,"df")
            for (root,dirs,files) in os.walk(df_path):
                for file_ in files:
                    if file_.endswith('.jpg'):
                        self.image_list.append(df_path+'/'+file_)
                        self.gt_list.append(1)

            real_path = os.path.join(self.image_dir,'real')
            for (root,dirs,files) in os.walk(real_path):
                for file_ in files:
                    if file_.endswith('.jpg'):
                        self.image_list.append(real_path+'/'+file_)
                        self.gt_list.append(0)
            
            #pdb.set_trace()

        elif split=='valid':
            self.transforms = T.Compose([T.Resize((256,256)),
                                        T.ToTensor()])
            self.image_dir = os.path.join(self.base_dir, 'validation')

            #list that holds full path to every image
            self.image_list = []
            #holds if real or fake
            self.gt_list = []

            df_path = os.path.join(self.image_dir,"df")
            for (root,dirs,files) in os.walk(df_path):
                for file_ in files:
                    if file_.endswith('.jpg'):
                        self.image_list.append(df_path+'/'+file_)
                        self.gt_list.append(1)

            real_path = os.path.join(self.image_dir,'real')
            for (root,dirs,files) in os.walk(real_path):
                for file_ in files:
                    if file_.endswith('.jpg'):
                        self.image_list.append(real_path+'/'+file_)
                        self.gt_list.append(0)

            #TO DO create self.image_list []
        elif split =='test':
            print('inference not implemented')
        
        



    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, index):
        '''returns data which is a dictionary that contains the image and the ground truth
        if image is fake or real'''

        fpath = self.image_list[index]
        image = Image.open(fpath)
        image = self.transforms(image)
        image = torch.FloatTensor(image)
        
        data = {}
        data['image'] = image
        data['ground_truth'] = self.gt_list[index]

        return data
