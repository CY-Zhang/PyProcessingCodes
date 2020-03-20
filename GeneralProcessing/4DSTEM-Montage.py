# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:37:15 2020

@author: Barnaby Levin

This code reshapes a 4D dataset into a giant 2D image of diffraction patterns
Data is first read from an HDF5 file in DE format, then downsampled to reduce the 
ultimate size of the 2D image. 
Finally, the 4D array is reshaped and the 2D montage is generated.

"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import pims as ps
from tifffile import imsave

Filename=r'C:\Users\Barnaby Levin\Documents\Microscope Data\4DSTEM\Harvard Test Data\25mx mag_256x48pix_2000fps_3.h5'

## Import HDF5 file (in DE format) and extract dataset
Dataset=h5.File(Filename,'r')
base_items=list(Dataset.items())
print('Items in Base Directory: ', base_items )
G1=Dataset.get('4DSTEM_experiment')
G1_items=list(G1.items())
print('Items in 4DSTEM_experiment: ', G1_items )
G2=G1.get('data')
G2_items=list(G2.items())
print('Items in data: ', G2_items )
G3=G2.get('datacubes')
G3_items=list(G3.items())
print('Items in datacubes: ', G3_items )
G4=G3.get('datacube_0')
G4_items=list(G4.items())
print('Items in datacube_0: ', G4_items )
G5=G4.get('data')

## Convert dataset to an array 
Data=np.array(G5)

## Before making the montage, we need to reduce the size of the diffraction patterns to something manageable, so rebin by 2
Datasize=np.shape(Data)
kxbin=np.floor_divide(Datasize[3],2)
kybin=np.floor_divide(Datasize[2],2)
DataBin2=np.zeros((Datasize[0]*Datasize[1],kybin,kxbin))
DataBin2=np.reshape(DataBin2, (Datasize[0],Datasize[1],kybin,kxbin))

for ki in range(0,kxbin):
    for kj in range(0,kybin):
        DataBin2[:,:,kj,ki]=np.divide(Data[:,:,2*kj,2*ki]+Data[:,:,2*kj+1,2*ki]+Data[:,:,2*kj,2*ki+1]+Data[:,:,2*kj+1,2*ki+1],4)

## Rebinning data by 2 may not be enough, so keep binning until the diffraction patterns reach a user defined Patternsize
Patternsize=16
Datasize2=np.shape(DataBin2)        
while Datasize2[3]>Patternsize:
    DataBig=np.copy(DataBin2)
    kxbin=np.floor_divide(Datasize2[3],2)
    kybin=np.floor_divide(Datasize2[2],2)
    DataBin2=np.zeros((Datasize2[0]*Datasize2[1],kybin,kxbin))
    DataBin2=np.reshape(DataBin2, (Datasize2[0],Datasize2[1],kybin,kxbin))
    for ki in range(0,kxbin):
        for kj in range(0,kybin):
            DataBin2[:,:,kj,ki]=np.divide(DataBig[:,:,2*kj,2*ki]+DataBig[:,:,2*kj+1,2*ki]+DataBig[:,:,2*kj,2*ki+1]+DataBig[:,:,2*kj+1,2*ki+1],4)
    Datasize2=np.shape(DataBin2)          
    
# Finally, reshape the 4D array into a Montage and plot the figure.
DataTranspose=np.transpose(DataBin2,(0,2,1,3))
Montage=np.reshape(DataTranspose,(Datasize[0]*kybin,Datasize[1]*kxbin))
Montage=np.float32(Montage)
plt.imshow(Montage, vmin=50, vmax=250)
plt.show()
