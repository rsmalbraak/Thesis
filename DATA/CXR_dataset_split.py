"""
Code and files to create the CXR dataset and rearrange the COVIDx V8A dataset 
into train and test sets which only contain images that originate from different 
datasets. See the thesis for an explanation on why we apply this new train/test split. 

Download the COVIDx V8A dataset using the images generating scrips at: 
https://github.com/lindawangg/COVID-Net
or download the dataset directly from: 
https://www.kaggle.com/datasets/andyczhao/covidx-cxr2

Downlaod the Shenzhen dataset from: 
https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities
For replication purposes, the following file contains the file names of the 270 
normal images rom the Shenzhen dataset used in this research:
normal_Shenzhen.txt
    
Note, a new and bigger version of the COVIDx V8A dataset has been uploaded that 
contains over 30.000 CXR images since the writing of this thesis. However, as we 
have posted all file names of the images present in our CXR dataset in the CXR_split.csv 
file, our dataset can still be retrieved by selecting the images in the CXR dataset 
with the file names from CXR_split.csv. 

CXR_split.csv contains the file names of the images together with the class_dataset 
labels which we use to divide the dataset

Code adapted from Bhanu Prathap (2020): 
https://github.com/bhanuprathap2000/dsp/blob/master/datasplit.py

To create our dataset, first move all downloaded images from both the train and 
and test set (and the subfolders) from the downloaded COVID V8A dataset into one 
folder "CXR", without any subfolders yet. Next, run this code to split all images 
in the CXR folder into subfolders based on their class labels and the dataset
that it originates from. For example, a pneumonia image from the Cohen dataset 
has label "pneumonia_cohen" in CXR_split.csv and will be saved in the corresponding 
subfolder. Next, rearrange the images in the generated subfolders to create the 
new train/test split by making the following folder directory and moving the 
corresponding images into the new folders: 
    
- TRAIN_3classes
    - COVID-19 pneumonia        : with images from the created subfolders COVID-19_sirm, COVID-19_actmed, COVID-19_fig1, COVID-19_ricord
    - non-COVID-19 pneumonia    : with images from the created subfolder pneumonia_rsna
    - normal                    : with images from the created subfolder normal_rsna
- TEST_3classes
    - COVID-19 pneumonia        : with images from the created subfolder COVID-19_cohen
    - non-COVID-19 pneumonia    : with images from the created subfolder pneumonia_cohen
    - normal                    : with images from the Shenzhen dataset (see below)

Note that the test set does not contain any normal images. Therefore, randomly 
select 270 normal images from the Shenzhen dataset and add those to  the 
TEST_3classes trainingset in the "normal" folder. 

-------------------------------------------------------------------------------
References datasets: 

COVIDx V8A dataset
Wang, L., Lin, Z. Q., & Wong, A. (2020). Covid-net: A tailored deep convolutional neural network
design for detection of covid-19 cases from chest x-ray images. Scientific Reports, 10 (1),
1–12

Shenzhen dataset:
National Library of Medicine, National Institutes of Health, Bethesda, MD, USA 
and Shenzhen No.3 People’s Hospital, Guangdong Medical College, Shenzhen, China

Jaeger S, Karargyris A, Candemir S, Folio L, Siegelman J, Callaghan F, Xue Z, 
Palaniappan K, Singh RK, Antani S, Thoma G, Wang YX, Lu PX, McDonald CJ. 
Automatic tuberculosis screening using chest radiographs. IEEE Trans Med Imaging. 
2014 Feb;33(2):233-45. doi: 10.1109/TMI.2013.2284099. PMID: 24108713

Candemir S, Jaeger S, Palaniappan K, Musco JP, Singh RK, Xue Z, Karargyris A, 
Antani S, Thoma G, McDonald CJ. Lung segmentation in chest radiographs using 
anatomical atlases with nonrigid registration. IEEE Trans Med Imaging. 
2014 Feb;33(2):577-90. doi: 10.1109/TMI.2013.2290491. PMID: 24239990

"""

#Load necessary pacakges
from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

###############################################################################

# First, we split the the COVID V8a dataset into subfolders that each contain a
# a different class / dataset label combination 

# Read the CSV file that contains the file names of the images together with 
# the class_dataset labels which we use to divide the dataset
labels = pd.read_csv(r"C:\Users\user\Documents\Masters Econometrics\Thesis\CXR_split.csv")

# Give the location of the dataset (the "CXR" folder) where all the images are present
# Next, create a folder DR in the same CXR folder as in which the images are present 
train_dir = r"C:\Users\user\Documents\Masters Econometrics\Thesis\CXR"
DR = r"C:\Users\user\Documents\Masters Econometrics\Thesis\CXR\DR"
if not os.path.exists(DR):
    os.mkdir(DR)

# Create directories and split the train set based on the dataset of origin and class label
for filename, class_name in labels.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(DR + str(class_name)):
        os.mkdir(DR + str(class_name))
    src_path = train_dir + '/'+ filename
    dst_path = DR + str(class_name) + '/' + filename
    try:
        shutil.move(src_path, dst_path)
        print("sucessful")
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))

##############################################################################
# Next, rearrange the images from the subfolders to create the following 
# new train/test split by making the following folder directory and moving the 
# corresponding images into the new folders: (and remove the COVID-19_sirm...
#pneumonia_cohen folders) 
#   
#- TRAIN_3classes
#    - COVID-19 pneumonia        : with images from the created subfolder COVID-19_sirm, COVID-19_actmed, COVID-19_fig1, COVID-19_ricord
#    - non-COVID-19 pneumonia    : with images from the created subfolder pneumonia_rsna
#    - normal                    : with images from the created subfolder normal_rsna
#- TEST_3classes
#    - COVID-19 pneumonia        : with images from the created subfolder COVID-19_cohen
#    - non-COVID-19 pneumonia    : with images from the created subfolder pneumonia_cohen
#    - normal                    : with images from the Shenzhen dataset (see below)
# - Randomly select 270 normal images from the Shenzhen dataset and inlude those
#   in the "normal" folder in TEST_3classes. 

################################################################################
