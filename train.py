import argparse
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from deepfake_data import deepfakeDataset
#import network
from torch.utils.tensorboard import SummaryWriter
from classifiers import Meso4
import os


def validate(model, test_dataset_loader, writer,loss_func, epoch):
    num_batches = len(test_dataset_loader)
    correct = 0
    count = 0
    total_loss = 0
    model.eval()
    for batch_idx, data in enumerate(test_dataset_loader):
        image = data['image'].cuda()
        ground_truth = data['ground_truth'].view((len(data['ground_truth']),1)).float()
        ground_truth = ground_truth.cuda()

        prediction = model(image)

        #calc loss
        loss = loss_func(prediction,ground_truth)
        total_loss +=loss.item()

        #calc accuracy
        for i in range(prediction.shape[0]):
            count+=1
            if (prediction[i]<0.2 and ground_truth[i]==0):
                correct+=1
            elif(prediction[i]>=0.8 and ground_truth[i]==1):
                correct+=1
    avg_loss = total_loss/count   
    print("Epoch: {}, Validation has loss {}".format(epoch, avg_loss))

    accuracy = correct/count
    #send to tensorboard
    writer.add_scalar('validation/accuracy', accuracy,epoch)
    print("Epoch: {}, Validation accuracy is {}".format(epoch, accuracy))




def main():

    #set up tensorboard
    writer = SummaryWriter()
    output_dir = 'saved_model'
    #load dataset
    #train_dataset = deepfakeDataset(split='train',image_dir=None)
    #test_dataset = deepfakeDataset(split='valid', image_dir=None)
    train_dataset = deepfakeDataset(split='train',image_dir='/home/ubuntu/VLR-16824/VLR-project/deepfake_database/deepfake_database')
    test_dataset = deepfakeDataset(split='valid', image_dir='/home/ubuntu/VLR-16824/VLR-project/deepfake_database/deepfake_database')

    #train_dataset_loader = torch.utils.data.DataLoader(train_dataset,batch_size=5, shuffle=True)
    #test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batchsize, shuffle=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)

    
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

    scheduler_adam = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=5, gamma=0.1)
    #scheduler_SGD = torch.optim.lr_scheduler.StepLR(optimizer_SGD, step_size=5, gamma=0.1)
    
    #loss function
    loss_func_mse = nn.MSELoss()

    #train
    step = 0
    for epoch in range(args.epochs):
        num_batches = len(train_dataset_loader)
        for batch_idx, data in enumerate(train_dataset_loader):
            #pdb.set_trace()
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
            loss.backward()
            #step optimizer
            optimizer_adam.step()

            #write to tensorboard
            writer.add_scalar('train/loss', loss.item(),current_step)

            #print
            if current_step % args.log_every ==0:
                print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_idx, num_batches, loss))
        
        validate(model, test_dataset_loader, writer,loss_func_mse, epoch)
    
        #step scheduler, steps based on set step size
        scheduler_adam.step()

        #To Do save model
        if epoch % 10 and current_step > 0:
            save_name = os.path.join(
            output_dir, '{}_{}.e10'.format(epoch,current_step))
            torch.save(model.state_dict(), save_name)
            print('Saved model to {}'.format(save_name))






def parse_args():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    parser = argparse.ArgumentParser(description='deepfake detection')
    # config for dataset
    parser.add_argument('--batchsize', type=int, default=1, metavar='N',
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
