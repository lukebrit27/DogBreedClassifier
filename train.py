import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torchvision.models as models
import os
from torchvision import datasets
from PIL import ImageFile
import torch.nn as nn
from glob import glob
from tqdm import tqdm
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
from workspace_utils import keep_awake
from train_funcs import *

data_dir = 'data/dog_images'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# check if CUDA is available
use_cuda = torch.cuda.is_available()


## Specify data loaders
train_transforms = transforms.Compose([transforms.RandomRotation(30), #rotate 30 degrees random direction
                                       #transforms.RandomResizedCrop(224),#randomly resize and take 224 centre crop
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],#means of each color channel
                                                            [0.229, 0.224, 0.225])])#sds of each color channel
valid_transforms = transforms.Compose([transforms.Resize(256),#resize 256x256 pixels(squares)
                                       transforms.CenterCrop(224),#crops square from center with 224 pixels on each side
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),#resize 256x256 pixels(squares)
                                       transforms.CenterCrop(224),#crops square from center with 224 pixels on each side
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

loaders_transfer = {'train':trainloader, 'valid':validloader, 'test':testloader}

# Load VGG16 model to use for tranfer learning
model_transfer = models.vgg16(pretrained=True)
for param in model_transfer.parameters():
    param.requires_grad = False


# Load our own classifier and attach it to the VGG16 model
model_transfer.classifier = Classifier()
for param in model_transfer.classifier.parameters():
    param.requires_grad = True

if use_cuda:
    model_transfer = model_transfer.cuda()


# specify learning rate, criterion and optimizer
learnrate = 0.001
criterion_transfer = nn.NLLLoss()
optimizer_transfer = optim.Adam(model_transfer.classifier.parameters(), lr=learnrate)


# train the model
# optimal model (i.e. model with minimum validation loss) will be saved as model_tranfer.pt state dict in the current directory
train(40, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')
