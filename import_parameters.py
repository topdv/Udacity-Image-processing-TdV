#This file will be used by train.py and predict.py

# import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import copy
import argparse

#Import for image processing
from PIL import Image
import glob, os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

#import for visualisation
import seaborn as sns

#import JSON label mapping
import json


#####Setting parameters

# Input for transforms
resize = 256
center_crop = 224
random_rotation = 30
random_resize = 224
network_means = [0.485, 0.456, 0.406]
network_stds = [0.229, 0.224, 0.225]
