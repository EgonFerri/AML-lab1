import numpy as np
from numpy import histogram as hist



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
#import sys, os, inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#filteringpath = os.path.join(parentdir, 'Filtering')
#sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # flattening the image to iterate over it
    flattened = [pix for dim in img_gray for pix in dim]

    min_interval = 0 
    max_interval = 255 
    
    # the bin size is equal to the difference of the extremes
    # divided by the input number of bins
    bin_size = (max_interval-min_interval)/num_bins
    bin_hist = {min_interval:0}
    
    # filling the dictionary's keys with the equal-distanced bins 
    previous = min_interval
    for i in range(num_bins):
        bin_ = previous + bin_size
        bin_hist[bin_] = 0
        previous = bin_
    
    # filling the dictionary's values with the frequencies of the 
    # pixels in the bins intervals
    keys = list(bin_hist.keys())
    for pix in flattened:
        for i in range(len(bin_hist)):
            if keys[i-1]<= pix < keys[i]:
                bin_hist[keys[i-1]] += 1

    hists = list(bin_hist.values())
    hists.pop()
    bins = list(bin_hist.keys())
    return  hists/np.sum(hists), np.array([round(bin_,3) for bin_ in bins])



#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    
    # flattening the image to iterate over it
    flattened = [pix for dim in img_color_double for pix in dim]
    
    min_interval = 0 
    max_interval = 255
    
    # the bin size is equal to the difference of the extremes
    # divided by the input number of bins
    bin_size = (max_interval-min_interval)/num_bins
    bin_hist = {min_interval:0}
    
    # filling the dictionary's keys with the equal-distanced bins 
    previous = min_interval
    for i in range(num_bins):
        bin_ = previous + bin_size
        bin_hist[bin_] = 0
        previous = bin_

    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    # Loop for each pixel i in the image 
    keys = list(bin_hist.keys())
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
            rgb = [0,0,0]
            for k in range(len(bin_hist)):
                if keys[k-1] <= flattened[i][0] < keys[k]:
                    rgb[0] = k-1
                if keys[k-1] <= flattened[i][1] < keys[k]:
                    rgb[1] = k-1
                if keys[k-1] <= flattened[i][2] < keys[k]:
                    rgb[2] = k-1
                    
            hists[rgb[0],rgb[1],rgb[2]] += 1

    #Normalize the histogram such that its integral (sum) is equal 1
    hists = hists/np.sum(hists)
    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    
    flattened = [pix for dim in img_color_double for pix in dim]
    min_interval = 0 
    max_interval = 255 
    
    # the bin size is equal to the difference of the extremes
    # divided by the input number of bins    
    bin_size = (max_interval-min_interval)/num_bins
    bin_hist = {min_interval:0}
    
    # filling the dictionary's keys with the equal-distanced bins 
    previous = min_interval
    for i in range(num_bins):
        bin_ = previous + bin_size
        bin_hist[bin_] = 0
        previous = bin_

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # filling the array's values with the frequencies of the 
    # pixels in the bins intervals
    keys = list(bin_hist.keys())
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G value of the pixel i
        rg = [0,0]
        for k in range(len(bin_hist)):
            if keys[k-1] <= flattened[i][0] < keys[k]:
                rg[0] = k-1
            if keys[k-1] <= flattened[i][1] < keys[k]:
                rg[1] = k-1
                    
        hists[rg[0],rg[1]] += 1

    #Normalize the histogram such that its integral (sum) is equal 1
    hists = hists/np.sum(hists)
    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    
    
    min_interval = -6
    max_interval = 6
    
    # defining the x and y derivatives of the image and
    # defining the minimum and maximum range of the derivatives
    derivx, derivy = gauss_module.gaussderiv(img_gray, 3)
    derivx = np.clip(derivx, min_interval, max_interval)
    derivy = np.clip(derivy, min_interval, max_interval) 
    
    # stacking the derivatives to iterate over them
    stacked = list(zip(derivx.reshape(-1), derivy.reshape(-1)))

    # the bin size is equal to the difference of the extremes
    # divided by the input number of bins    
    bin_size = (max_interval - min_interval)/num_bins
    
    # filling the list's values with the equal-distanced bins
    bins = [min_interval for _ in range(num_bins+1)]
    previous = min_interval
    for i in range(num_bins):
        bin_ = previous + bin_size
        bins[i+1] = bin_
        previous = bin_

    # defining a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # filling the array's values with the frequencies of the 
    # pixels in the bins intervals
    for i in range(len(stacked)):

        deriv_xy = [0,0]
        for k in range(len(bins)):
            if bins[k-1] <= stacked[i][0] < bins[k]:
                deriv_xy[0] = k-1
            if bins[k-1] <= stacked[i][1] < bins[k]:
                deriv_xy[1] = k-1
                    
        hists[deriv_xy[0],deriv_xy[1]] += 1
    
    hists = hists/np.sum(hists)
    # return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists


def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

