"""
Code for the classification of COVID-19, non-COVID-19 pneumonia and normal classes 
using three different CNNs (SqueezeNet, ResNet-18, AlexNet) using deep transfer learning 
via fine tuning. The models are trained on the augmented dataset by HAC-GAN. 

Our code inspired by and adapted from:  
https://github.com/madsendennis/notebooks/blob/master/pytorch/3_PyTorch_Transfer_learning.ipynb


"""
# Import necessary packages
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation
import copy
import random
import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import matplotlib.pyplot as plt # for plotting

# Set random seed for reproducibility
manualSeed = 100
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

output_dir = r'/home/rsmalbraak/output_cnn/all/'

# Set the number of epochs
num_epochs = 25

# Applying Transforms to the Data
# Different transforms are used for each CNN

#Transforms for SqueezeNet
image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=227),
        transforms.CenterCrop(size=227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=227),
        transforms.CenterCrop(size=227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=227),
        transforms.CenterCrop(size=227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#Transforms for ResNet-18
image_resnet = { 
    'train': transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#Transforms for AlexNet
image_alexnet = { 
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Set train and test directory paths
train_directory = r'/home/rsmalbraak/data/All/TRAIN_3classes'
# Set directory with generated images by HAC-GAN to augment the original CXR dataset
train_directory2= r'/home/rsmalbraak/data/All/output'
test_directory = r'/home/rsmalbraak/data/All/TEST_3classes'

# Batch size 
bs = 64
validation_split = .2
shuffle_dataset = True

# Number of classes
num_classes = len(os.listdir(train_directory))  #10#2#257
print(num_classes)

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'train_gen':datasets.ImageFolder(root=train_directory2, transform=image_transforms['train']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

data_resnet = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_resnet['train']),
    'train_gen':datasets.ImageFolder(root=train_directory2, transform=image_resnet['train']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_resnet['test'])
}

data_alexnet = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_alexnet['train']),
    'train_gen':datasets.ImageFolder(root=train_directory2, transform=image_alexnet['train']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_alexnet['test'])
}

# Augment the original datasets with the fake images generated by HAC-GAN
data['concat'] = torch.utils.data.ConcatDataset([data['train'], data['train_gen']])
data_resnet['concat'] = torch.utils.data.ConcatDataset([data_resnet['train'], data_resnet['train_gen']])
data_alexnet['concat'] = torch.utils.data.ConcatDataset([data_alexnet['train'], data_alexnet['train_gen']])

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['concat'])

# Creating data indices for training and validation split
indices = list(range(train_data_size))
split = int(np.floor(validation_split * train_data_size))
print(split)
valid_data_size = split
if shuffle_dataset :
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders for training and validation:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_data_loader = DataLoader(data['concat'], batch_size=bs , sampler= train_sampler,num_workers=12)
valid_data_loader = DataLoader(data['concat'], batch_size=bs, sampler = valid_sampler,num_workers=12)
test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True,num_workers=12)

train_data_loader_resnet = DataLoader(data_resnet['concat'], batch_size=bs , sampler= train_sampler,num_workers=12)
valid_data_loader_resnet = DataLoader(data_resnet['concat'], batch_size=bs, sampler = valid_sampler,num_workers=12)
test_data_loader_resnet = DataLoader(data_resnet['test'], batch_size=bs, shuffle=True,num_workers=12)

train_data_loader_alexnet = DataLoader(data_alexnet['concat'], batch_size=bs , sampler= train_sampler,num_workers=12)
valid_data_loader_alexnet = DataLoader(data_alexnet['concat'], batch_size=bs, sampler = valid_sampler,num_workers=12)
test_data_loader_alexnet = DataLoader(data_alexnet['test'], batch_size=bs, shuffle=True,num_workers=12)

def train_and_validate(model, loss_criterion, optimizer, epochs):
    '''
    Function to train and validate SqueezeNet
    model : is the model to train and validate
    loss_criterioin : is the loss criterion to minimize
    optimizer : is the optimizer for computing the gradients and updating the weights
    epochs : is the number of epochs
    

    Returns the trained model with best validation loss
    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start = time.time()
    min_valid_loss = np.inf

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        # Iterate over the batches 
        for i, (inputs, labels) in enumerate(train_data_loader,0):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Compute the  mean accuracy
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
        # Validation 
        with torch.no_grad():

            # Set to evaluation mode as the weights are not updated with validation
            model.eval()

            # Iterate over the validation batches
            for j, (inputs, labels) in enumerate(valid_data_loader):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                                
                 # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                          
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size
        
        #If the validation loss for the current epoch is lower than the lowest validation loss 
        #up to this epoch, we save the current model as the best model
        if min_valid_loss > avg_valid_loss:
            print("Epoch best loss")
            print(epoch)
            min_valid_loss = avg_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # Load the best model weights
        model.load_state_dict(best_model_wts)
        
        # Save the parameters of the best model
        torch.save({
        'squeez_state_dict': model.state_dict(),
        }, output_dir +"squeezenet_final" + '.tar') 
    
       
    return model

def train_and_validate_resnet(model, loss_criterion, optimizer, epochs):
    '''
    Function to train and validate ResNet18
    model : is the model to train and validate
    loss_criterioin : is the loss criterion to minimize
    optimizer : is the optimizer for computing the gradients and updating the weights
    epochs : is the number of epochs
    

    Returns the trained model with best validation loss
    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start = time.time()
    min_valid_loss = np.inf

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        # Iterate over the batches 
        for i, (inputs, labels) in enumerate(train_data_loader_resnet,0):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Compute the  mean accuracy
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
        # Validation 
        with torch.no_grad():

            # Set to evaluation mode as the weights are not updated with validation
            model.eval()

            # Iterate over the validation batches
            for j, (inputs, labels) in enumerate(valid_data_loader_resnet):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                                
                 # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                          
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        #If the validation loss for the current epoch is lower than the lowest validation loss 
        #up to this epoch, we save the current model as the best model
        if min_valid_loss > avg_valid_loss:
            print("Epoch best loss")
            print(epoch)
            min_valid_loss = avg_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # Load the best model weights
        model.load_state_dict(best_model_wts)
        
        # Save the parameters of the best model
        torch.save({
        'resnet_state_dict': model.state_dict(),
        }, output_dir +"resnetnet_final" + '.tar') 
    
       
    return model

def train_and_validate_alexnet(model, loss_criterion, optimizer, epochs):
    '''
    Function to train and validate AlexNet
    model : is the model to train and validate
    loss_criterioin : is the loss criterion to minimize
    optimizer : is the optimizer for computing the gradients and updating the weights
    epochs : is the number of epochs
    

    Returns the trained model with best validation loss
    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start = time.time()
    min_valid_loss = np.inf

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        # Iterate over the batches 
        for i, (inputs, labels) in enumerate(train_data_loader_alexnet,0):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Compute the  mean accuracy
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
        # Validation 
        with torch.no_grad():

            # Set to evaluation mode as the weights are not updated with validation
            model.eval()

            # Iterate over the validation batches
            for j, (inputs, labels) in enumerate(valid_data_loader_alexnet):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                                
                 # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                          
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size
        
        #If the validation loss for the current epoch is lower than the lowest validation loss 
        #up to this epoch, we save the current model as the best model
        if min_valid_loss > avg_valid_loss:
            print("Epoch best loss")
            print(epoch)
            min_valid_loss = avg_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # Load the best model weights
        model.load_state_dict(best_model_wts)
        
        # Save the parameters of the best model
        torch.save({
            'alex_state_dict': model.state_dict(),
            }, output_dir +"alexnet_final" + '.tar') 
    
       
    return model

# Set the device            
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################
#Prepare SqueezeNet for deep transfer learning

#Load the pre-trained SqueezeNet 
squeezenet = models.squeezenet1_1(pretrained=True)

# Change the final convolutional layer of SqueezeNet for Deep Transfer Learning, 
# such that it outputs a prediction over the three classes in the CXR dataset
squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
squeezenet = squeezenet.to(device)

# Define Optimizer and Loss Function
loss_func_squeez = nn.CrossEntropyLoss()
optimizer = optim.Adam(squeezenet.parameters())

##############################################################################
#Prepare ResNet-18 for deep transfer learning

#Load the pre-trained ResNet-18 
model_conv = models.resnet18(pretrained=True)

# Change the final fully connected layer of ResNet-18 for Deep Transfer Learning, 
# such that it outputs a prediction over the three classes in the CXR dataset
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)
model_conv = model_conv.to(device)

# Define Optimizer and Loss Function
loss_func_res = nn.CrossEntropyLoss()
optimizer_res = optim.Adam(model_conv.parameters())

##############################################################################
#Prepare AlexNet for deep transfer learning

#Load the pre-trained AlexNet
alexnet = models.alexnet(pretrained=True)
print(alexnet)

# Change the final fully connected (or linear) layer of AlexNet for Deep Transfer Learning, 
# such that it outputs a prediction over the three classes in the CXR dataset
num_ftrs = alexnet.classifier[6].in_features
alexnet.classifier[6] = nn.Linear(num_ftrs, num_classes)
alexnet = alexnet.to(device)

# Define Optimizer and Loss Function
loss_func_lex = nn.CrossEntropyLoss()
optimizer_lex = optim.Adam(alexnet.parameters())

###############################################################################

# Train the pre-trained SqueezeNet on the augmented CXR dataset
trained_model_squeez = train_and_validate(squeezenet, loss_func_squeez, optimizer, num_epochs)
trained_model_squeez.eval()

# Print the confusion matrices and accuracies of SqueezeNet after training
correct = 0
correct_norm = 0
correct_cov = 0
correct_pneum = 0
total = 0
confusion_matrix = torch.zeros(num_classes, num_classes)
print(confusion_matrix)
#{0: 'COVID-19 pneumonia', 1: 'non-COVID-19 pneumonia', 2: 'normal'}
with torch.no_grad():
    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = trained_model_squeez(images)
        _, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
           confusion_matrix[t.long(), p.long()] += 1
        
print('Accuracy of squeezenet on the test images: %d %%' % (
    100 * correct / total))
print(confusion_matrix)
print(confusion_matrix[0,0]/torch.sum(confusion_matrix[0,]))
print(confusion_matrix[1,1]/torch.sum(confusion_matrix[1,]))
print(confusion_matrix[2,2]/torch.sum(confusion_matrix[2,]))


# Train the pre-trained ResNet-18 on the augmented CXR dataset
trained_model_res = train_and_validate_resnet(model_conv, loss_func_res, optimizer_res, num_epochs)
trained_model_res.eval()

# Print the confusion matrices and accuracies of ResNet-18 after training
correct = 0
correct_norm = 0
correct_cov = 0
correct_pneum = 0
total = 0
confusion_matrix = torch.zeros(num_classes, num_classes)
#{0: 'COVID-19 pneumonia', 1: 'non-COVID-19 pneumonia', 2: 'normal'}
with torch.no_grad():
    for images, labels in test_data_loader_resnet:
        images = images.to(device)
        labels = labels.to(device)
               
        outputs = trained_model_res(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        

print('Accuracy of resnet on the test images: %d %%' % (
    100 * correct / total))
print(confusion_matrix)
print(confusion_matrix[0,0]/torch.sum(confusion_matrix[0,]))
print(confusion_matrix[1,1]/torch.sum(confusion_matrix[1,]))
print(confusion_matrix[2,2]/torch.sum(confusion_matrix[2,]))

# Train the pre-trained AlexNet on the augmented CXR dataset
trained_model_alex = train_and_validate_alexnet(alexnet, loss_func_lex, optimizer_lex, num_epochs)
trained_model_alex.eval()

# Print the confusion matrices and accuracies of AlexNet after training
correct = 0
correct_norm = 0
correct_cov = 0
correct_pneum = 0
total = 0
confusion_matrix = torch.zeros(num_classes, num_classes)
#{0: 'COVID-19 pneumonia', 1: 'non-COVID-19 pneumonia', 2: 'normal'}
with torch.no_grad():
    for images, labels in test_data_loader_alexnet:
        images = images.to(device)
        labels = labels.to(device)
               
        outputs = trained_model_alex(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print('Accuracy of alexnet on the test images: %d %%' % (
    100 * correct / total))
print(confusion_matrix)
print(confusion_matrix[0,0]/torch.sum(confusion_matrix[0,]))
print(confusion_matrix[1,1]/torch.sum(confusion_matrix[1,]))
print(confusion_matrix[2,2]/torch.sum(confusion_matrix[2,]))
