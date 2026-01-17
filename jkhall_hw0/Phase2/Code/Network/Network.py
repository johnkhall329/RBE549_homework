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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = F.cross_entropy(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc, pred = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc, 'pred': pred, 'labels': labels}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def test_epoch_end(self, TestSet, TestLabels):
        out = self(TestSet)
        acc, pred = accuracy(out, TestLabels)
        return acc.item(), pred

    def batch_end(self, batch, result):
        print("MiniBatch [{}], loss: {:.4f}, acc: {:.4f}".format(batch, result['loss'], result['acc']))

    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))



class UpdatedCIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(16)

        self._to_linear = None
        x_rand = torch.randn(3,32,32).view(-1,3,32,32)
        self.convs(x_rand)

        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, OutputSize)

    def convs(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.pool(F.relu(x))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        x = self.convs(xb)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ResNetCIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.maxpool = nn.MaxPool2d((2,2))

        # Residual Block 1
        self.res1 = nn.Conv2d(6, 12, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 12, 3, padding=1)

        # Residual Block 2
        self.res2 = nn.Conv2d(12, 24, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv5 = nn.Conv2d(24, 24, 3, padding=1)

        self.avgpool = nn.AvgPool2d(5)

        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2_1 = nn.BatchNorm2d(12)
        self.batch_norm2_2 = nn.BatchNorm2d(12)
        self.batch_norm3_1 = nn.BatchNorm2d(24)
        self.batch_norm3_2 = nn.BatchNorm2d(24)

        self._to_linear = None
        x_rand = torch.randn(3,32,32).view(-1,3,32,32)
        self.convs(x_rand)

        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, OutputSize)

    def convs(self, x):
        x = self.conv1(x)
        x = self.maxpool(F.relu(self.batch_norm1(x)))

        residual = self.res1(x)
        x = self.conv2(x)
        x = F.relu(self.batch_norm2_1(x))
        x = self.conv3(x)
        x = self.batch_norm2_2(x)
        x += residual
        x = F.relu(x)

        residual = self.res2(x)
        x = self.conv4(x)
        x = F.relu(self.batch_norm3_1(x))
        x = self.conv5(x)
        x = self.batch_norm3_2(x)
        x += residual
        x = self.avgpool(F.relu(x))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        x = self.convs(xb)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ResNeXtCIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize, cardinality=4):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.maxpool = nn.MaxPool2d((2,2))

        # ResNext Group 1
        self.resnext_blocks1 = ResNeXtBlock(8, 16, bottleneck_width=4, cardinality=cardinality)

        # Residual Block 2
        self.resnext_blocks2 = ResNeXtBlock(16, 32, bottleneck_width=4, cardinality=cardinality)

        self.avgpool = nn.AvgPool2d(5)

        self.batch_norm1 = nn.BatchNorm2d(8)

        self._to_linear = None
        x_rand = torch.randn(3,32,32).view(-1,3,32,32)
        self.convs(x_rand)

        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, OutputSize)

    def convs(self, x):
        x = self.conv1(x)
        x = self.maxpool(F.relu(self.batch_norm1(x)))

        x = self.resnext_blocks1(x)
        x = self.resnext_blocks2(x)
        x = self.avgpool(F.relu(x))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        x = self.convs(xb)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_width, cardinality):
        super().__init__()

        group_width = in_channels * bottleneck_width
        self.res1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(group_width, out_channels, kernel_size=1, bias=False)

        self.batch1 = nn.BatchNorm2d(group_width)
        self.batch2 = nn.BatchNorm2d(group_width)
        self.batch3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.res1(x)
        x = self.conv1(x)
        x = F.relu(self.batch1(x))
        x = self.conv2(x)
        x = F.relu(self.batch2(x))
        x = self.conv3(x)
        x = self.batch3(x)
        x += residual
        x = F.relu(x)
        return x
    
class DenseNetCIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize, compression=0.5, growth_rate=8):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        
        in_channels = 32
        layer_count = 3
        self.conv1 = nn.Conv2d(3, in_channels, 5, padding=2, bias=False)
        self.batch1 = nn.BatchNorm2d(in_channels)
        self.maxpool = nn.MaxPool2d((2,2))

        self.block1 = DenseBlock(in_channels=in_channels, layer_count=layer_count, growth_rate=growth_rate)
        self.trans1 = TransitionLayer(in_channels=in_channels + layer_count*growth_rate, compression=compression)

        self.block2 = DenseBlock(in_channels=int((in_channels + layer_count*growth_rate)*compression), layer_count=layer_count, growth_rate=growth_rate)
        self.trans2 = TransitionLayer(in_channels=int((in_channels + layer_count*growth_rate)*compression) + layer_count*growth_rate, compression=compression)

        self.block3 = DenseBlock(in_channels=int((int((in_channels + layer_count*growth_rate)*compression) + layer_count*growth_rate)*compression), layer_count=layer_count, growth_rate=growth_rate)

        self.batch2 = nn.BatchNorm2d(int((int((in_channels + layer_count*growth_rate)*compression) + layer_count*growth_rate)*compression) + layer_count*growth_rate)
        self.avgpool = nn.AvgPool2d(2)

        self._to_linear = None
        x_rand = torch.randn(3,32,32).view(-1,3,32,32)
        self.convs(x_rand)

        self.fc1 = nn.Linear(self._to_linear, 96)
        self.fc2 = nn.Linear(96, OutputSize)

    def convs(self, x):
        x = self.conv1(x)
        x = self.maxpool(F.relu(self.batch1(x)))

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.avgpool(F.relu(x))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        x = self.convs(xb)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        inter_channels = in_channels * 4 # Bottleneck
        self.res1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.batch1 = nn.BatchNorm2d(in_channels)
        self.batch2 = nn.BatchNorm2d(inter_channels)

    def forward(self, x):
        input = x
        x = F.relu(self.batch1(x))
        x = self.conv1(x)
        x = F.relu(self.batch2(x))
        x = self.conv2(x)
        x = torch.cat([x, input], 1)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, layer_count, growth_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(layer_count):
            self.layers.append(DenseLayer(in_channels + i*growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression):
        super().__init__()

        self.batch = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, int(in_channels * compression), kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.batch(x))
        x = self.conv(x)
        x = self.avgpool(x)
        return x

class InitialCIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self._to_linear = None
        x_rand = torch.randn(3,32,32).view(-1,3,32,32)
        self.convs(x_rand)

        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, OutputSize)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        x = self.convs(xb)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

