# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):   
    sig2=sigma**2
    
    x=list(map(round,np.arange(-3*(sig2), 3*(sig2)+1))) #define the range
    
    Gx= [np.exp(-0.5*((el*el)/sig2)) for el in x] #find gaussian values
    Gx= Gx/sum(Gx) #normalize
    
    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma): 
    
    [Gx, pos] = gauss(sigma) #get gaussian kernel
    
    smooth_img=np.apply_along_axis(func1d=lambda x: np.convolve(x, Gx, "same"),axis=0, arr=img) #apply to all rows
    smooth_img=np.apply_along_axis(func1d=lambda x: np.convolve(x, Gx, "same"),axis=1, arr=smooth_img) #apply to all columns

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    sig2=sigma**2
    
    x=list(map(round,np.arange(-3*(sig2), 3*(sig2)+1))) #define the range
    
    Dx= [-(el/sig2) * (np.exp(-0.5*((el*el)/sig2))) for el in x] #find derivative gaussian values
    Dx= Dx/sum(Dx) #normalize
    
    return Dx, x



def gaussderiv(img, sigma):

    [Dx, pos] = gaussdx(sigma) #get gaussian kernel
    
    imgDx=np.apply_along_axis(func1d=lambda x: np.convolve(x, Dx, "same"),axis=0, arr=img) #apply to all rows
    imgDy=np.apply_along_axis(func1d=lambda x: np.convolve(x, Dx, "same"),axis=1, arr=img) #apply to all columns


    
    return imgDx, imgDy

