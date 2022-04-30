"""
Implementation of our customized version of ACGAN, which we base on the DCGAN architecture.  
Compared to standard implementations, we add Gaussian layers to the discriminator, use 
dropout in both D and G, we use multiplication of the noise and class vectors.
Also, we transform the DCGAN architecture such that is operates as an AC-GAN. 

See the thesis for an in-depth explanation and motivation of the architecture 
and all used modules. 

References code: 
Inspired by the following DCGAN implementation from the paper Unsupervised Representation 
Learning with Deep Convolutional Generative Adversarial Networks (https://arxiv.org/abs/1511.06434)
https://github.com/pytorch/examples
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

"""
#Import necessary packages
from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Set seed for reproducibility
manualSeed = 100
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#Set hyperparameters
batch_size = 64
image_size = 128
nc = 3 #number of channels in the input images
nz = 100 #dimension latent vector and embedding vector
ngf = 64 #sets the depth of feature maps propagated through the generator
ndf = 64 #sets the depth of feature maps propagated through the discriminator
ne = 1000 #number of epochs
lr_g = 0.0002 #learning rate generator
lr_d = 0.0002 #learning rate discriminator
b1 = 0.5 #first parameter adam optimizer
ngpu = 1 #number of GPUs
nw = 12 #number of workers of the GPU
num_class = 3 #number of classes
std = 0.1 #standard deviation used in Gaussian layer
std_decay_rate = 0.00012  #decay rate used in Gaussian layer

#Paths to the datasets
train_path = r'/home/rsmalbraak/data/All/TRAIN_3classes'
output_dir = r'/home/rsmalbraak/output/'

#Set up the device 
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Weights initialization for generator and discriminator from the DCGAN paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Gaussian noise module that adds noise to the inputs of the discriminator
# From https://github.com/ShivamShrirao/facegan_pytorch
class GaussianNoise(nn.Module):                    
    def __init__(self, std, decay_rate):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

#Function to decay the standard deviation of GaussianNoise
#From https://github.com/ShivamShrirao/facegan_pytorch
std_list =[]
def decay_gauss_std(net):
    std = 0
    for m in net.modules():
        if isinstance(m, GaussianNoise):
            m.decay_step()
            std_list.append(m.std)

#Define transformations for the train and test sets
transform_train = transforms.Compose([transforms.Resize((image_size,image_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
                                       ])

#Create the datasets
train_tf = datasets.ImageFolder(root=train_path,transform=transform_train)
#create the dataloader, which returns batches of images and the corresponding labels
trainloader_tf = torch.utils.data.DataLoader(train_tf,batch_size=batch_size,
                                             shuffle=True,num_workers = nw)

##############################################################################
# Generator

"""
Contains blocks B1,...,B6 (see thesis for in-depth explanation of all used modules). 

Takes a random noise vector and class label as input, and outputs a 128x128x3 image. 

"""
############################################################################## 

class Generator(nn.Module):
    
    def __init__(self, ngpu,nz,ngf,nc,num_class):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.num_class = num_class
                
        #Embedding layer
        self.embedding = nn.Embedding(num_class,nz)
                     
        # input 100 x 1 x 1 
        # output 1024 x 4 x 4 
        self.B1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = nz, 
                               out_channels = ngf * 16,
                               kernel_size = 4, 
                               stride = 1, 
                               padding=0, 
                               bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.Dropout2d(0.3))
        
        # input 1024 x 4 x 4
        # output 512 x 8 x 8         
        self.B2 =  nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf * 16, 
                               out_channels = ngf * 8, 
                               kernel_size = 4,
                               stride=2,
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Dropout2d(0.3))
        
        # input 512 x 8 x8 
        # output 256 x 16 x 16
        self.B3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf * 8, 
                               out_channels = ngf * 4, 
                               kernel_size = 4, 
                               stride=2,
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Dropout2d(0.3))
        
        # input 256 x 16 x 16
        # output 128 x 32 x 32        
        self.B4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf * 4, 
                               out_channels = ngf * 2, 
                               kernel_size = 4, 
                               stride=2,
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Dropout2d(0.3))    
    
        # input 128 x 32 x 32
        # output 64 x 64 x 64
        self.B5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf * 2,
                               out_channels = ngf, 
                               kernel_size = 4, 
                               stride=2,
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Dropout2d(0.3)) 
        
        # input 64 x 64 x 64
        # output 3 x 128 x 128
        self.B6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf,
                               out_channels = nc, 
                               kernel_size = 4,
                               stride=2, 
                               padding = 1, 
                               bias=False),
            nn.Tanh()) 
        
    def forward(self, noise, label):
        
        #return the class embedding
        label_embedding = self.embedding(label)
        #multiplicate noise and embedding vector
        x = torch.mul(noise,label_embedding)
        x = x.view(-1,nz,1,1)
        
        #forward the tensor through blocks B1,...,B6 to generate a synthetic image        
        x = self.B1(x)
        x = self.B2(x)
        x = self.B3(x)
        x = self.B4(x)
        x = self.B5(x)
        x = self.B6(x)
        
        return x
 
##############################################################################
# Discriminator 
"""
Contains blocks B1,...,B7 (see thesis for in-depth explanation of all used modules). 

Takes an image as input, and outputs the source and class predicted probabilities. 

"""
##############################################################################
    
class Discriminator(nn.Module):
    def __init__(self,ngpu,nc,ndf,std,std_decay_rate):
        
    #def __init__(self,ndf,nc,num_class):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.std = std
        self.std_decay_rate = std_decay_rate
                   
        # input 3 x 128 x 128
        # output 64 x 64 x 64        
        self.B1 = nn.Sequential(
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Conv2d(in_channels = nc , 
                      out_channels = ndf, 
                      kernel_size = 4,
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3))
        
        # input 64 x 64 x 64  
        # output 128 x 32 x 32
        self.B2 = nn.Sequential(
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Conv2d(in_channels = ndf, 
                      out_channels = ndf * 2, 
                      kernel_size = 4, 
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf * 2),           
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3))
        
        
        # input 128 x 32 x 32
        # output 256 x 16 x 16
        self.B3 = nn.Sequential(
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Conv2d(in_channels = ndf * 2, 
                      out_channels = ndf * 4, 
                      kernel_size = 4, 
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3))
        
        
        # input 256 x 16 x 16
        # output 512 x 8 x 8 
        self.B4 = nn.Sequential( 
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Conv2d(in_channels = ndf * 4, 
                      out_channels = ndf * 8, 
                      kernel_size = 4,
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3))
        
    
        # input 512 x 8 x 8 
        # output 1024 x 4 x 4 
        self.B5 = nn.Sequential(
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Conv2d(in_channels = ndf * 8, 
                      out_channels = ndf * 16, 
                      kernel_size = 4,
                      stride=2, 
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3))
        
        
        #Source prediction
        # input 1024 x 4 x 4 
        # output 1 x 1 x 1 
        self.B6 = nn.Sequential(
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Conv2d(in_channels = ndf * 16,
                      out_channels = 1,
                      kernel_size = 4, 
                      stride=1, 
                      padding=0,
                      bias = False),
            nn.Sigmoid())
        
        #Class prediction
        # input 1024 x 4 x 4 
        # output 3 x 1 x 1 
        self.B7 = nn.Sequential(
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Conv2d(in_channels = ndf * 16,
                      out_channels = num_class,
                      kernel_size = 4, 
                      stride=1, 
                      padding=0, 
                      bias = False),
            nn.LogSoftmax(dim = 1))
                        
    def forward(self, x):
        
        #Forward the images through block B1,...,B7 to output a source 
        #prediction and predicted probability distribution over the classes
        x = self.B1(x)
        x = self.B2(x)
        x = self.B3(x)
        x = self.B4(x)
        x = self.B5(x)
        p_source = self.B6(x)
        p_class = self.B7(x)
        
        p_source = p_source.view(-1)
        p_class = p_class.view(-1,num_class)
    
        return p_source, p_class 
    
##############################################################################
# Training 
##############################################################################

if __name__ == '__main__':
    
    # Create the generator
    G = Generator(ngpu,nz,ngf,nc,num_class).to(device)

    # Apply the weight initialization function to randomly initialize all weights
    G.apply(weights_init)
   
    # Create the Discriminator
    D = Discriminator(ngpu,nc,ndf,std,std_decay_rate).to(device)

    # Apply the weights initialization function to randomly initialize all weights
    D.apply(weights_init)

    #Initialize loss functions
    criterion_S = nn.BCELoss() # Source loss function
    criterion_C = nn.NLLLoss() # Class loss function

    # Use latent vectors to visualize the progression of the generator 
    fixed_noise = torch.randn(64, nz, device=device)
    label_sampled2 = torch.randint(0,num_class,(64,),
                                          device = device, dtype = torch.long)
    
    # Setup Adam optimizers for the discriminator D and generator G
    optim_D = optim.Adam(D.parameters(), lr=lr_d, betas=(b1, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=lr_g, betas=(b1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_x_value = []
    D_G_z1_value = []
    D_G_z2_value = []
   
    G_losses10 = []
    D_losses10 = []
    D_x_value10 = []
    D_G_z1_value10 = []
    D_G_z2_value10 = []
    D_source = []
    G_source = []
    iters = 0
        
    # Start the training loop
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(ne):
        print("Epoch:")
        print(epoch)
        # For each batch in the dataloader
        for i, data in enumerate(trainloader_tf, 0):
            
            #-------------------------------------------------------------------------
            #Training the discriminator
            #-------------------------------------------------------------------------
            
            image, label = data
            image = image.to(device)
            label = label.to(device)
            batch_size = image.size(0)
            
            # Establish convention for real and fake labels during training
            real_label = torch.FloatTensor(batch_size).to(device)
            real_label.fill_(1)
            fake_label = torch.FloatTensor(batch_size).to(device)
            fake_label.fill_(0)
            
            #Use one sided label smoothing 
            soft_label = torch.Tensor(batch_size).uniform_(0.9, 1).to(device)
                        
            # Train with real data
            #------------------------------------------------------------------
            optim_D.zero_grad()
                                 
            #Forward batch of real images into discriminator 
            output_source, output_class = D(image)        
            
            #Calculate "real" losses
            error_Dsource = criterion_S(output_source,soft_label)
            error_Dclass = criterion_C(output_class,label)            
            error_Dreal = error_Dsource + error_Dclass
            
            #Calculate gradients for D in the backward pass
            error_Dreal.backward()
            
            D_x = output_source.mean().item()  
        
            # Train with fake data, first generate fake data
            #------------------------------------------------------------------
            
            #Generate batch of latent vectors and sample class labels
            noise = torch.randn(batch_size, nz, device=device)
            label_sampled = torch.randint(0,num_class,(batch_size,),
                                      device = device, dtype = torch.long)
            
            #Generate fake image batch with the generator G
            fake_image = G(noise,label_sampled)
                                
            # Forward batch of fake images into discriminator and calculate "fake" losses
            # Use detach as we will use reuse fake image later
            output_source, output_class = D(fake_image.detach())
            
            #Calculate "fake" losses
            error_DFsource = criterion_S(output_source,fake_label)
            error_DFclass = criterion_C(output_class,label_sampled)            
            error_Dfake = error_DFsource + error_DFclass
            
            #Calculate gradients for D in the backward pass, accumulated with the previous gradients
            error_Dfake.backward()
            
            D_G_z1 = output_source.mean().item()
        
            #Compute error of D as sum over the real and fake batches
            errorD = error_Dreal + error_Dfake
            
            #Update the parameters of the discriminator
            optim_D.step()
            
            #Compute source loss of D as sum over the real and fake batches
            loss_D = error_DFsource + error_Dsource
            
        #-------------------------------------------------------------------------
        #Training the Generator
        #-------------------------------------------------------------------------
            
            # Establish convention for real and fake labels during training
            # For generator, the labels are seen as real, so all equal 1
            val_label = torch.FloatTensor(batch_size).to(device)
            val_label.fill_(1)
            
            optim_G.zero_grad()            
            
            #Perform another forward pass of fake batches through D
            output_source, output_class = D(fake_image)
            
            #Calculate the generator loss
            errorG_source = criterion_S(output_source,val_label)
            errorG_class = criterion_C(output_class,label_sampled)
            errorG = errorG_source + errorG_class
            
            #Calculate gradients for G in the backward pass
            errorG.backward()
            
            D_G_z2 = output_source.mean().item()
            
            #Update the parameters of the generator
            optim_G.step()        
                      
            # Save Losses for plotting
            G_losses.append(errorG.item())
            D_losses.append(errorD.item())
            D_x_value.append(D_x)
            D_G_z1_value.append(D_G_z1)
            D_G_z2_value.append(D_G_z2)
            D_source.append(loss_D.item())
            G_source.append(errorG_source.item())

        #After each epoch, decay the standard deviation used in the Gaussian layer
        decay_gauss_std(D)
       
        # Check how the generator is doing by saving G's output on fixed_noise       
        with torch.no_grad():
            fake = G(fixed_noise,label_sampled2).detach().to(device)
            fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
            vutils.save_image(fake_grid, output_dir + "_images" + str(epoch) + ".png")
           
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_dir + "plots_loss_DG" + ".png")  
    
    plt.figure(figsize=(10,5))
    plt.title("D(x) and D(G(z)) Loss During Training")
    plt.plot(D_x_value,label="D(x)")
    plt.plot(D_G_z1_value,label="D(G(z))")
    plt.xlabel("iterations")
    plt.ylabel("Discriminator values")
    plt.legend()
    plt.savefig(output_dir + "plots_d_x_d_gz" + ".png")
    
    #Save the model, use it to generate CXRs later
    torch.save({
        'netG_state_dict': G.state_dict(),
        'netD_state_dict': D.state_dict(),
        'optimizerG_state_dict': optim_G.state_dict(),
        'optimizerD_state_dict': optim_D.state_dict()
       }, output_dir +"model_final" + '.tar') 
