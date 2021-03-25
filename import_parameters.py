#This file will be used by train.py and predict.py

######Importing dependancies
print ('importing')

import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import copy

parser = argparse.ArgumentParser(description = 'All arguments to be used in predicting and training')
parser.add_argument ('--train_dir', default = 'flowers/train',  help = 'Directory of training fotos', metavar = '')
parser.add_argument ('--image_path', default = "flowers/test/10/image_07090.jpg",  help = 'Image path of immage to be classified', metavar = '')
parser.add_argument ('--checkpoint', default = "checkpoint.pth", help = 'Path to checkpoint to be used', metavar = '')
args = parser.parse_args()


#Import for image processing
from PIL import Image
import glob, os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

#import for visualisation
import seaborn as sns

#Label mapping
import json


Lr = 0.001


#####Setting parameters

#Setting directories
data_dir = 'flowers'
train_dir = args.train_dir #data_dir + '/train' 
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



# Define your transforms for the training, validation, and testing sets

# Input for transforms
resize = 256
center_crop = 224
random_rotation = 30
random_resize = 224
network_means = [0.485, 0.456, 0.406]
network_stds = [0.229, 0.224, 0.225]



#Define classifier size and learning rate
layers = [25088, 1024, 200, 102]


data_transforms = {
    'train': transforms.Compose([
                                transforms.RandomRotation(random_rotation), 
                                transforms.RandomResizedCrop(random_resize), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor(),
                                transforms.Normalize(network_means, network_stds)   ]),
   
    'validate' : transforms.Compose([
                                transforms.Resize(resize), 
                                transforms.CenterCrop(center_crop),
                                transforms.ToTensor(),
                                transforms.Normalize(network_means, network_stds)   ]),
   
    'test' : transforms.Compose([
                                transforms.Resize(resize),
                                transforms.CenterCrop(center_crop),
                                transforms.ToTensor(),
                                transforms.Normalize(network_means, network_stds)  ])
}


# Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform= data_transforms['train']),
    'validate': datasets.ImageFolder(valid_dir, transform = data_transforms['validate']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}





# Define the dataloaders using the image datasets and the trainforms
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
    'validate': torch.utils.data.DataLoader(image_datasets['validate'], batch_size = 32),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32)
}   


#import JSON label mapping
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

    
## define model architecture
#Define model
model = models.vgg16(pretrained=True)

# don't compute gradients
for param in model.parameters():
    param.requires_grad = False


    
    
# Making sure all training is done on the external GPU if this is available. Else the calculations are done using the CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)




