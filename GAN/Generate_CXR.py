"""
Code that can be used to generate CXR after training AC-GAN or HAC-GAN. 

"""
#import necessary packages
from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
# Load the generator of eiter AC-GAN or HAC-GAN to generate the images
#from AC-GAN import Generator as G
from HAC_GAN import Generator as G
from torchvision.utils import save_image

#Set hyperparameters
batch_size = 64
image_size = 128
nc = 3 #number of channels in the input images
nz = 100 #dimension latent vector and embedding vector
ngf = 64 #sets the depth of feature maps propagated through the generator
ndf = 64 #sets the depth of feature maps propagated through the discriminator
lr_g = 0.0002 #learning rate generator
lr_d = 0.0002 #learning rate discriminator
b1 = 0.5 #first parameter adam optimizer
ngpu = 0 #number of GPUs
nw = 1 #number of workers of the GPU
num_class = 3 #number of classes

#Set up device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Load the saved and trained generator of AC-GAN or HAC-GAN
FILE = "C:/Users/user/Documents/Masters Econometrics/Thesis/output/model_final.tar"
OUTPUT = "C:/Users/user/Documents/Masters Econometrics/Thesis/CXR_dataset/output/"
#Set up the generator of AC-GAN or HAC-GAN to generate the images
loaded_model = G(ngpu,nz,ngf,nc,num_class)
#Load the saved parameters/weights after training into the generator of AC-GAN or HAC-GAN
loaded_model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu'))["netG_state_dict"])
loaded_model.eval()

#Generate COVID-19 pneumonia images
for i in range(10000-2088):
    
    #Sample noise and the label "0" that corresponds to the COVID-19 pneumonia class
    fixed_noise = torch.randn(1, nz, device=device)
    label_sampled = torch.tensor([0], dtype = torch.long)

    #Generate images by forwarding noise and label through the trained generator G
    image = loaded_model(fixed_noise,label_sampled)
    
    #Save the generated image
    path = OUTPUT+'COVID-19 pneumonia/cov_' + str(i) + ".png"
    save_image(image,path,normalize=True)
    
#Generate non-COVID-19 pneumoina iamges
for i in range(10000-5523):
    
    #Sample noise and the label "1" that corresponds to the non-COVID-19 pneumonia class
    fixed_noise = torch.randn(1, nz, device=device)
    label_sampled = torch.tensor([1], dtype = torch.long)

    #Generate images by forwarding noise and label through the trained generator G
    image = loaded_model(fixed_noise,label_sampled)
    
    #Save the generated image
    path =  OUTPUT+'non-COVID-19 pneumonia/pneum_' + str(i) + ".png"
    save_image(image,path,normalize=True)
    
#Generate normal images
for i in range(10000-8066):

    #Sample noise and the label "2" that corresponds to the COVID-19 pneumonia class
    fixed_noise = torch.randn(1, nz, device=device)
    label_sampled = torch.tensor([2], dtype = torch.long)

    #Generate images by forwarding noise and label through the trained generator G
    image = loaded_model(fixed_noise,label_sampled)
    
    #Save the generated images
    path = OUTPUT+'normal/norm_' + str(i) + ".png"
    save_image(image,path,normalize=True)
   
    
    
 
