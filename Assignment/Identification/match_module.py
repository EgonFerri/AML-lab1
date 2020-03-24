# Libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module # functions for histograms
import gauss_module # functions for gaussian filters
import dist_module # functions for compute distances

import glob # to load list of files
import pandas as pd
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import ParameterGrid # useful in implementig the grid search



# Functions

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain
#       handles to distance and histogram functions, and to find out whether histogram function
#       expects grayvalue or color image
# note: use functions compute_histograms to get from a list of file names of images a list of
#       corresponding histograms

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    # whether histogram function expects grayvalue or color image
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

    # form list of file images to list of histograms
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images))) # container for distances
    best_match = [] # container for best match for each image

    for i in range(len(query_images)):
        Q = query_hists[i]
        for j in range(len(model_hists)):
            V = model_hists[j]
            h1 = Q.copy()
            h2 = V.copy()
            dist = dist_module.get_dist_by_name(h1, h2, dist_type) # distance between histogram
            D[j, i] = dist # save distance measure in correct position
        idx = list(D[:, i]).index(min(D[:, i])) # find histogram with lower distance respect to the one in exam
        best_match.append(idx)

    return np.array(best_match), D



# Map each image in the list to the specified corresponding histogram and put the resulting histogram in a list
#
# image_list - list which specifies file names of images
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
# hist_isgray - boolean which specifies whether histogram function expects grayvalue or color image
#
# note: use function single_hist_maker to get for each image the corresponding histogram

def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

    image_hist = list(map(lambda x: single_hist_maker(x, hist_type, hist_isgray, num_bins), np.array(image_list)))

    return image_hist



# Given an image file it returns the specified corresponing histogram
#
# img - file name of an image
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
# hist_isgray - boolean which specifies whether histogram function expects grayvalue or color image

def single_hist_maker(img, hist_type, hist_isgray, num_bins):
    # Gray or colored image
    if hist_isgray == True:
        img = rgb2gray(np.array(Image.open(img)))
    else:
        img = np.array(Image.open(img))
    # Histogram type
    hist = histogram_module.get_hist_by_name(np.array(img, float), num_bins, hist_type)
    if hist_type == 'grayvalue':
        return hist[0]
    else:
        return hist


# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

    num_nearest = 5  # show the top-5 neighbors
    best_match, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

    for i in range(len(query_images)):

        # sort distances of images and retrive index of the num_nearest (i.e. 5) with lower distances
        closest = sorted(range(len(D[:, i])), key=lambda k: D[:, i][k])[:num_nearest]

        # Plot
        for j in range(1, num_nearest+2):

            # general specifications for plot
            plt.figure(num_nearest+1, figsize=(20, 20))
            # subplot command to show num_nearest best images in the same row
            ax = plt.subplot(1, num_nearest+1, j)
            ax.title.set_size(25)
            plt.axis('off')
            plt.sca(ax)

            if j == 1: # the first image is the query image
                ax.title.set_text('Q'+str(i))
                plt.imshow(np.array(Image.open(query_images[i])))
            else:
                # plot best matches with corresponding distances
                ax.title.set_text('M'+str(round(D[closest[j-2], i], 2)))
                plt.imshow(np.array(Image.open(model_images[closest[j-2]])))
        plt.show()

    return None



# Given a distance measure and a histogram type compute over a set of query image the recognition rate.
# The recognition rate is is given by a ratio between number of correct matches and total number of query images.
#
# model_images - list of file names of model images
# query_images - list of file names of some query images
# all_query - list of file names of all query images (it is needed to check wheter the match is correct or not)
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use the previously implemented function 'find_best_match' to find best match.

def recognition_rate(model_images, query_images, all_query, dist_type, hist_type, num_bins):
    best_match, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    num_correct = sum( np.array(query_images) == np.array(all_query)[[best_match]])
    return num_correct / len(query_images)



# Find recognition rate for each combination of distance/histogram function and number of bins
#
# hyper_grid - dictionaty in which key -> hyperparameters, value -> list of values for that hyperparameter
# model_images - list of file names of model images
# query_images - list of file names of some query images
# all_query - list of file names of all query images (it is needed to check wheter the match is correct or not)

def grid_search(hyper_grid, model_images, query_images, all_query):

    grid = ParameterGrid(hyper_grid)
    all_res = []
    for hypers in tqdm(grid):

        # Combinations
        dist_type = hypers['dist_type']
        hist_type = hypers['hist_type']
        num_bins = hypers['num_bins']
        hyperparams = [dist_type, hist_type, num_bins]

        # Compute recognition rate
        rate = recognition_rate(model_images, query_images, all_query, dist_type, hist_type, num_bins)

        # Save results
        all_res.append({'hyperparameters': hyperparams,'recognition_rate': rate})

    out = pd.DataFrame(all_res)
    out.to_csv('recognition_rates.csv', index=False)

    return print('File created grid_search')
