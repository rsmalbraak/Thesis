# -*- coding: utf-8 -*-
"""

Code to compute the MS-SSIMM score between two sets of images. 

Adopted from Pessoa, J. (2021). Pytorch differentiable multi-scale structural similarity (ms-ssim) loss. 
https://github.com/jorge-pessoa/pytorch-msssim
https://github.com/VainF/pytorch-msssim

"""
# import necessary packages
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from pytorch_msssim import ms_ssim

# Set random seed
manualSeed = 100
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# image size needs to be >160 due to 4 downsampling operations in MSSSIM
imgsize = 256

# Define transforms to resize the images
image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=imgsize),
       transforms.CenterCrop(size=imgsize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])}

# Define directory to load the data
train_directory = "C:/Users/user/Documents/Masters Econometrics/Thesis/CXR_dataset/output"
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train'])
   }

# We randomly sample 100 pairs of images in a dataset by employing the train/validation split 
# We sample 100 images from the train set and 100 images from the validation set
# Creating data indices for training and validation splits:
bs = 100
validation_split = .2
shuffle_dataset = True
train_data_size = len(data['train'])
indices = list(range(train_data_size))
split = int(np.floor(validation_split * train_data_size))
if shuffle_dataset :
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_data_loader = DataLoader(data['train'], batch_size=bs , sampler= train_sampler)
train_data_loader2 = DataLoader(data['train'], batch_size=bs , sampler= valid_sampler)

# Get a batch of 100 images from both the train and validation set
X , label = next(iter(train_data_loader))
Y , label = next(iter(train_data_loader2))

# calculate ms-ssim between the batches, we normalized images to the range of the TanH function [-1,1] so set datarange to 2
ms_ssim= ms_ssim(X, Y, data_range=2, size_average=True ) 
print(ms_ssim)
