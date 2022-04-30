# -*- coding: utf-8 -*-
"""
Code to compute the FID score between two datasets. 

Adapted from https://github.com/GaParmar/clean-fid

Parmar, G., Zhang, R., & Zhu, J.-Y. (2022). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. In CVPR.

"""

# import necessary packages
from cleanfid import fid

# Define two directories/folders with two datasets, between which we want to calculate the FID score
fdir1 = r'/home/rsmalbraak/data/TRAIN_3classses'
fdir2 = r'/home/rsmalbraak/data/output'

# Compute FID between two image folders
score = fid.compute_fid(fdir1, fdir2, mode = "clean")
print(score)
