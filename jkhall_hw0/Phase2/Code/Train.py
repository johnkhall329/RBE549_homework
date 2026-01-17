#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


from sklearn.metrics import confusion_matrix
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
# from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
# import Misc.ImageUtils as iu
from Network.Network import InitialCIFAR10Model, UpdatedCIFAR10Model, ResNetCIFAR10Model, ResNeXtCIFAR10Model, DenseNetCIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *



# Don't generate pyc codes
sys.dont_write_bytecode = True

def StandardizeInputs(data):
    x_min = data.min()
    x_max = data.max()
    # Handle the case where all values are the same to avoid division by zero
    if x_min == x_max:
        data = torch.zeros_like(data)
    else:
        data = 2 * (data - x_min) / (x_max - x_min) - 1
    
    return data

def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1)
        
        ImageNum += 1

        I1, Label = TrainSet[RandIdx]

        # Append All Images and Mask
        I1Batch.append(StandardizeInputs(I1))
        # I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, TestSet, TestLabels, LogsPath):
    """
    Inputs: 
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    TestSet - The test dataset
    TestLabels - Labels corresponding to Test
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Initialize the model
    model = DenseNetCIFAR10Model(InputSize=3*32*32,OutputSize=10) 

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = AdamW(model.parameters(), lr=1e-3)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath +'/' + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
    
    test_set = torch.from_numpy(TestSet.data).to(torch.float32)
    test_set = test_set.permute(0,3,1,2)
    test_set = StandardizeInputs(test_set)
    test_labels = torch.tensor(TestSet.targets)
        
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        batch_results = []
        epoch_preds = []
        epoch_labels = []
        if Epochs % 10 == 0 and Epochs != 0: # Decrease learning rate
            for g in Optimizer.param_groups:
                print(f'Learning rate decreased from {g["lr"]:.5f} to {(g["lr"] * 0.9):.5f}')
                g['lr'] = g['lr'] * 0.9
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)
            
            # Predict output with forward pass
            LossThisBatch = model.training_step(Batch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            
            result = model.validation_step(Batch)
            batch_results.append(result)
            epoch_preds += result['pred'].tolist()
            epoch_labels += result['labels'].tolist()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath + '/' + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                model.batch_end(Epochs*NumIterationsPerEpoch + PerEpochCounter, result)
                torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
                # print('\n' + SaveName + ' Model Saved...')

        # Save model every epoch
        SaveName = CheckPointPath + '/' + str(Epochs) + 'model.ckpt'
        epoch_results = model.validation_epoch_end(batch_results)

        with torch.no_grad():
            epoch_test_results = model.test_epoch_end(test_set, test_labels)

        # Tensorboard
        Writer.add_scalar('LossEveryEpoch', epoch_results["loss"], Epochs)
        Writer.add_scalar('TrainAccuracy', epoch_results["acc"], Epochs)
        Writer.add_scalar('TestAccuracy', epoch_test_results[0], Epochs)
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()
        model.epoch_end(Epochs, epoch_results)
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        
        print('\n' + SaveName + ' Model Saved...')

    model.eval()
    with torch.no_grad():
        epoch_test_results = model.test_epoch_end(test_set, test_labels)

    train_cm = confusion_matrix(y_true=epoch_labels,  # True class for test-set.
                                y_pred=epoch_preds,
                                normalize='pred')  # Predicted class.
    
    PlotConfusionMatrix(train_cm, TrainSet.classes, "DenseNet Training", normalize=True)

    test_cm = confusion_matrix(y_true=TestSet.targets,  # True class for test-set.
                               y_pred=epoch_test_results[1],
                               normalize='pred')  # Predicted class.
    
    PlotConfusionMatrix(test_cm, TrainSet.classes, "DenseNet Test", normalize=True)

    num_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad: num_params += param.numel()
    print('Number of parameters in this model are %d ' % num_params)

    onnx = torch.onnx.export(model, torch.randn(1, 3, 32, 32), CheckPointPath + '/' + 'FinalModel.onnx', dynamo=True)
    Writer.add_graph(model, torch.randn(1, 3, 32, 32))
    Writer.flush()
    Writer.close()

    torch.save(model.state_dict(), CheckPointPath + '/' + 'FinalModel.pt')
        

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/DenseNet', help='Path to save Logs for Tensorboard, Default=Logs/')
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=ToTensor())
    TestSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=ToTensor())

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    BasePath = os.path.abspath(os.path.dirname(__file__))
    CheckPointPath = os.path.abspath(os.path.join(BasePath, CheckPointPath))
    
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, TestLabels, NumClasses = SetupAll(BasePath, CheckPointPath)


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, TestSet, TestLabels,LogsPath)

    
if __name__ == '__main__':
    main()
 
