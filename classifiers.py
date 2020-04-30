import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

IMGWIDTH = 256

class Meso4(nn.Module):

	def __init__(self, num_classes=, inp_size=, c_dim=3,dl_rate=):
		super().__init__()
		self.num_classes = num_classes

		#what about input dimension?
		#keep the image 256
		self.conv1 = nn.Conv2d(c_dim,8,3,strides=1,padding=2)
		
		self.conv2 = nn.Conv2d(8,8,5,strides=1,padding=2)
		
		self.conv3 = nn.Conv2d(8,16,5,strides=1,padding=2)
		self.conv4 = nn.Conv2d(16,16,5,strides=1,padding=2)

		self.nonlinear = lambda x:nn.functional.relu(x)		
		self.batchNorm = nn.BatchNorm2d(c_dim)
		
		self.pool1 = nn.MaxPool2d(2, 2)
		self.pool2 = nn.MaxPool2d(4, 4)
		
		self.flat_dim = 
		
		self.leakyrelu = nn.LeakyReLU(0.1)

		self.dropout = nn.Dropout(p=0.5)

		self.sigmoid = lambda x:nn.functional.sigmoid(x)
		#what is dense? I think it is Linear
		#what is output dim?
		self.fc1 = nn.Linear(self.flat_dim,16)

		self.fc2 = nn.Linear(16,1)





	def forward(self,x):

		'''
			x = (N,C,H,W)

		'''

		N = x.size(0)
		x = self.conv1(x)
		x = self.nonlinear(x)
		x = self.batchNorm(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.nonlinear(x)
		x = self.batchNorm(x)
		x = self.pool1(x)

		x = self.conv3(x)
		x = self.nonlinear(x)
		x = self.batchNorm(x)
		x = self.pool1(x)

		x = self.conv4(x)
		x =self.nonlinear(x)
		x = self.batchNorm(x)
		x= self.pool2(x)

		x = x.view(N,self.flat_dim)

		x = self.dropout(x)

		x = self.fc1(x)

		x = self.leakyrelu(x)

		x= self.dropout(x)

		x= self.fc2(x)

		x = self.sigmoid(x)

		return x



class MesoInception4(nn.Module):
	pass
