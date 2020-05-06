import argparse
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from deepfake_data import deepfakeDataset
from torch.utils.tensorboard import SummaryWriter
from classifiers import Meso4, MesoInception4
import os
import matplotlib.pyplot as plt



def main():
    if args.person=='Aubrey':
        test_dataset = deepfakeDataset(split='test', image_dir='/home/aubs/VLR-project2/real-and-fake-face-detection')
        test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)
        model_path = '/home/aubs/VLR-project2/VLR-project/saved_model/Aubrey_7/44_10124.e10' #inception
        model_path = '/home/aubs/VLR-project2/VLR-project/saved_model/Aubrey_6/44_7424.e10' #regular Meso

    #load model
    model = Meso4()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    num_batches = len(test_dataset_loader)
    correct = 0
    count = 0
    total_loss = 0
    model.eval()
    for batch_idx, data in enumerate(test_dataset_loader):
        image = data['image'].cuda()
        ground_truth = data['ground_truth'].view((len(data['ground_truth']),1)).float()
        ground_truth = ground_truth.cuda()

        prediction, heatmapout = model(image)

        #calc accuracy
        for i in range(prediction.shape[0]):
            count+=1
            if (prediction[i]<0.2 and ground_truth[i]==0):
                correct+=1
            elif(prediction[i]>=0.8 and ground_truth[i]==1):
                correct+=1

    accuracy = correct/count
    #send to tensorboard
    print("Test Accuracy is {}".format(accuracy))




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
    parser.add_argument('--person',
                        help='who is running the code')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    return args
    


if __name__ == '__main__':
    args = parse_args()
    main()

