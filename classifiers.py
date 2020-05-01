import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pdb

IMGWIDTH = 256

class Meso4(nn.Module):

	def __init__(self, c_dim=3):
		super().__init__()

		#what about input dimension?
		#keep the image 256
		self.conv1 = nn.Conv2d(c_dim,8,3,stride=1,padding=1)
		
		self.conv2 = nn.Conv2d(8,8,5,stride=1,padding=2)
		
		self.conv3 = nn.Conv2d(8,16,5,stride=1,padding=2)
		self.conv4 = nn.Conv2d(16,16,5,stride=1,padding=2)

		self.nonlinear = lambda x:nn.functional.relu(x)		
		self.batchNorm1 = nn.BatchNorm2d(8)
		self.batchNorm2 = nn.BatchNorm2d(8)
		self.batchNorm3 = nn.BatchNorm2d(16)
		self.batchNorm4 = nn.BatchNorm2d(16)

		self.pool1 = nn.MaxPool2d(2, 2)
		self.pool2 = nn.MaxPool2d(4, 4)
		
		self.flat_dim = 1024
		
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
		x = self.batchNorm1(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.nonlinear(x)
		x = self.batchNorm2(x)
		x = self.pool1(x)
		pdb.set_trace()

		x = self.conv3(x)
		x = self.nonlinear(x)
		x = self.batchNorm3(x)
		x = self.pool1(x)

		x = self.conv4(x)
		x =self.nonlinear(x)
		x = self.batchNorm4(x)
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
