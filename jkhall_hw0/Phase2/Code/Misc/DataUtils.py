#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Prof. Nitin J. Sanket (nsanket@wpi.edu)
Assistant Professor,
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import itertools
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath, CheckPointPath):
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    DirNamesTrain =  SetupDirNames(BasePath)

    # Read and Setup Labels
    LabelsPath = os.path.abspath(os.path.join(BasePath, 'TxtFiles'))
    TrainLabels = ReadLabels(LabelsPath+'/LabelsTrain.txt')
    TestLabels = ReadLabels(LabelsPath+'/LabelsTest.txt')

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       print(CheckPointPath)
       os.makedirs(CheckPointPath)
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    ImageSize = [32, 32, 3]
    NumTrainSamples = len(DirNamesTrain)

    # Number of classes
    NumClasses = 10

    return DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, TestLabels, NumClasses

def ReadLabels(LabelsPathTrain):
    if(not (os.path.isfile(LabelsPathTrain))):
        print('ERROR: Train Labels do not exist in '+LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, 'r')
        TrainLabels = TrainLabels.read()
        TrainLabels = map(float, TrainLabels.split())

    return TrainLabels
    

def SetupDirNames(BasePath): 
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesPath = os.path.abspath(os.path.join(BasePath, 'TxtFiles/DirNamesTrain.txt'))
    DirNamesTrain = ReadDirNames(DirNamesPath)        
    
    return DirNamesTrain

def ReadDirNames(ReadPath):
    """
    Inputs: 
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames

def PlotConfusionMatrix(cm, classes, title, normalize=False):
    figure = plt.figure(figsize=(len(classes), len(classes)))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title = f'Normalized {title} Confusion Matrix' if normalize else f'{title} Confusion Matrix'
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Use white text if square is dark, otherwise black.
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 3),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()