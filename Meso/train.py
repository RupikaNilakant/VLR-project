import argparse
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

#To DO: import model

def validate(model, test_dataset, writer):
    model.eval()
    for batch_idx, data in enumerate(test_dataset):
        image = data['image']
        ground_truth = data['ground_truth']
        prediction = model.(image)

        #calc loss
        #calc accuracy




def main():

    #set up tensorboard
    writer = SummaryWriter()

    #load dataset
    train_dataset = torch.utils.data.DataLoader()
    test_dataset = torch.utils.data.DataLoader()

    #define model
    #TO DO import model
    #model = ??

    #optimizer
    #I have multiple options we can try 
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    
    #scheduler

    #loss function
    loss_func_mse = nn.MSELoss()

    #train
    for epoch in range(5):
        for batch_idx, data in enumerate(train_dataset):
            model.train()
            image = data['image']
            ground_truth = data['ground_truth']
            prediction = model(image)

            loss = loss_func_mse(prediction,ground_truth)





    


if __name__ == '__main__':
    args, device = parse_args()
    main()