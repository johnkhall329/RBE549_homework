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

import itertools
import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
import math as m
import sklearn
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
from Network.Network import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img
    
def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred, classes):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred,
                          normalize='all')  # Predicted class.
    
    PlotConfusionMatrix(cm, classes, normalize=True)

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')


def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = CIFAR10Model(InputSize=3*32*32,OutputSize=10) 
    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
    
    OutSaveT = open(LabelsPathPred, 'w')

    for count in tqdm(range(len(TestSet))): 
        Img, Label = TestSet[count]
        # Img, ImgOrg = ReadImages(Img)
        PredT = torch.argmax(model(Img.view([-1, 3, 32, 32]))).item()

        OutSaveT.write(str(PredT)+'\n')
    OutSaveT.close()

       
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/aa/144model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    CheckPointPath = Args.CheckPointPath
    ModelPath = "C:\\Users\\johnk\\Documents\\RBE\\RBE549\\RBE549_homework\\jkhall_hw0\\Phase2\\Checkpoints\\4model.ckpt"
    LabelsPath = Args.LabelsPath
    # TestSet = CIFAR10(root='data/', train=False)
    TestSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=ToTensor())

    BasePath = os.path.abspath(os.path.dirname(__file__))
    CheckPointPath = os.path.abspath(os.path.join(BasePath, CheckPointPath))
    LabelsPath = os.path.abspath(os.path.join(BasePath, LabelsPath))
    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = os.path.abspath(os.path.join(BasePath, 'TxtFiles/PredOut.txt')) # Path to save predicted labels

    TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred, TestSet.classes)
     
if __name__ == '__main__':
    main()
 
