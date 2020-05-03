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
		self.conv1 = nn.Conv2d(c_dim,8,3,stride=1,padding=2)
		
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


class Inception(nn.Module):
	def __init__(self,a,b,c,d):
		super(Inception,self).__init__()

		self.conva = nn.Conv2d(inp_dim,a,1,padding=2)
		self.convb1 = nn.Conv2d(inp_dim,b,1,padding=2)
		self.convb3 = nn.Conv2d(inp_dim,b,3,padding=2)
		self.convc1 = nn.Conv2d(inp_dim,c,1,padding=2)
		self.convc3 = nn.Conv2d(inp_dim,c,3,padding=2,dilation=2)
		self.convd1 = nn.Conv2d(inp_dim,d,1,padding=2)
		self.convd3 = nn.Conv2d(inp_dim,d,3,padding=2,dilation=3)
		self.nonlinear = lambda x:nn.functional.relu(x)		

	def forward(self,x):

		x1 = self.conva(x)
		x1 = self.nonlinear(x1)

		x2 = self.convb1(x)
		x2 = self.nonlinear(x2)
		x2 = self.convb3(x2)
		x2 = self.nonlinear(x2)

		x3 = self.convc1(x)
		x3 = self.nonlinear(x3)
		x3 = self.convc3(x3)
		x3 = self.nonlinear(x3)

		x4 = self.convd1(x)
		x4 = self.nonlinear(x4)
		x4 = self.convd3(x4)
		x4 = self.nonlinear(x4)

		y = torch.cat([x1,x2,x3,x4],1)

		return y

class MesoInception4(nn.Module):
    def __init__(self, c_dim=3):
        super(MesoInception4,self).__init__()
    	
    	self.batchNorm1 = nn.BatchNorm2d(8)
    	self.pool1 = nn.MaxPool2d(2, 2)
		self.pool2 = nn.MaxPool2d(4, 4)
		self.flat_dim = 
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.dropout = nn.Dropout(p=0.5)
		self.sigmoid = lambda x:nn.functional.sigmoid(x)
		self.fc1 = nn.Linear(self.flat_dim,16)
		self.fc2 = nn.Linear(16,1)

		self.conv = nn.Conv2d(dim,16,5,padding=2)
		self.nonlinear = lambda x:nn.functional.relu(x)	
		self.incept1 = Inception(1,4,4,2)
		self.incept2 = Inception(2,4,4,2)

	def forward(self,x):

		N = x.size(0)

		x = self.incept1(x)
		x = self.batchNorm1(x)
		x = self.pool1(x)

		x = self.incept2(x)
		x = self.batchNorm1(x)
		x = self.pool1(x)

		x = self.conv(x)
		x = self.nonlinear(x)
		x = self.batchNorm1(x)
		x = self.pool1(x)

		x = self.conv(x)
		x = self.nonlinear(x)
		x = self.batchNorm1(x)
		x = self.pool2(x)

		x = x.view(N,self.flat_dim)

		x = self.dropout(x)
		x = self.fc1(x)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.sigmoid(x)

		return x



        
