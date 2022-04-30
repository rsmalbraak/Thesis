"""
Code to create Grad-CAM visualizations of the decision process of CNNs, imports and uses grad_cam.py.

We modify the original implementation and load the weights and architectures of the DTL CNNs trained on the CXR dataset
to visualize the decision process from the CNNs. 

Adapted code from https://github.com/kazuto1011/grad-cam-pytorch to load our own trained CNNs on the CXR dataset. 

"""

# Import necessary packages
from __future__ import print_function
import copy
import os.path as osp
import torch.nn as nn
import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import torch.optim as optim

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

#Set the device
def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device

#Function to load the images
def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

# Function to get class indices and class names
def get_classtable():
    classes = []
    with open("C:/Users/user/Documents/synset_words1.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes

# Function to preprocess the image, resize and normalize
def preprocess(image_paths):
    raw_image = cv2.imread(image_paths)
    raw_image = cv2.resize(raw_image, (224,224))
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

# Function to save the gradient
def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))

# Function to save gradcam visualization
def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))
    
# Following lines are from the original implementation, but we refrain from using those 
# as we load the modified trained versions of the CNNs trained on the CXR dataset

# # torchvision models
# model_names = sorted(
#     name
#     for name in models.__dict__
#     if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
# )


def demo1(image_paths, target_layer, topk, output_dir):
    """
    Generates the Grad-CAM visualizations. 
    
    We load our trained models (SqueezeNet, ResNet-18 or AlexNet adjusted for detecting COVID-19) with their learned weights 
    
    """
    # Set device
    cpu = 1
    device = get_device(cpu)

    # get classes based on the syset file
    classes = get_classtable()
    
    # Set directory to the final weights of our pretrained SqueezeNet on the CXR dataset 
    FILE = "C:/Users/user/Documents/models/squeezenet_final.tar"
    
    # Load the adapted SqueezeNet and the final weights after training on the CXR dataset
    # Set the model to evaluation mode
    model = models.squeezenet1_1(pretrained=True)    
    model.classifier[1] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu'))["squeez_state_dict"])
    model.to(device)
    model.eval()
      
    # Load images    
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")
    
    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({})".format(j, "model", classes[ids[j, i]], probs[j, i]))
           
            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "model", target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

          

# Set paths to the images
path1 = ""
path2 = ""

# Set image paths
image_paths = [path1, path2]
# Set target layer, whose feature maps are used to create the activation maps
target_layer = "features"
# Set the number of classes in the dataset
topk = 3
# Set directory to save the output of the feature maps
output_dir = "C:/Users/user/Documents/Masters Econometrics/Thesis/activation_map"
demo1(image_paths, target_layer, topk, output_dir)
    
    
