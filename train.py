import argparse
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from deepfake_data import deepfakeDataset
from classifiers import Meso4

#To DO: import model

def validate(model, test_dataset, writer):
    model.eval()
    for batch_idx, data in enumerate(test_dataset_loader):
        image = data['image']
        ground_truth = data['ground_truth']
        prediction = model(image)

        #calc loss
        #calc accuracy




def main():

    #set up tensorboard
    #writer = SummaryWriter()

    #load dataset
    train_dataset = deepfakeDataset(split='train',image_dir=None)
    test_dataset = deepfakeDataset(split='valid', image_dir=None)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,args.batch_size, shuffle=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=True)

    #define model
    #TO DO import model
    model = Meso4().cuda()

    #optimizer
    #I have multiple options we can try 
    
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    
    #scheduler 
    #step size relates to number of epoch
    #in the paper they say they step every 1000 iterations
    '''
    scheduler_adam = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=5, gamma=0.1)
    scheduler_SGD = torch.optim.lr_scheduler.StepLR(optimizer_SGD, step_size=5, gamma=0.1)
    '''
    #loss function
    loss_func_mse = nn.MSELoss()

    #train
    step = 0
    for epoch in range(args.epochs):
        num_batches = len(train_dataset_loader)
        for batch_idx, data in enumerate(train_dataset_loader):
            pdb.set_trace()
            current_step = epoch*num_batches + batch_idx
            #reset optimizer
            
            optimizer_adam.zero_grad()
            
            #set model to train
            
            model.train()
            
            #get data from dataloader
            image = data['image'].cuda()
            ground_truth = data['ground_truth'].view((len(data['ground_truth']),1)).float()
            ground_truth = ground_truth.cuda()
            #run through model and get prediction
            
            prediction = model(image)
            
            #calculate loss
            loss = loss_func_mse(prediction,ground_truth)
            #backprop
            loss.backwards()
            #step optimizer
            optimizer_adam.step()


            if current_step % args.log_every ==0:
                print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_idx, num_batches, loss))
        '''
        #step scheduler, steps based on set step size
        scheduler_adam.setp()
        '''
        #To Do save model








def parse_args():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    parser = argparse.ArgumentParser(description='deepfake detection')
    # config for dataset
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before evaluating model')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    return args
    


if __name__ == '__main__':
    args = parse_args()
    main()