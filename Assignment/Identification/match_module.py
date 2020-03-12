import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

import glob
import warnings
warnings.filterwarnings("ignore")

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

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images)))

    best_match = []

    for i in range(len(query_hists)):
        Q = query_hists[i]
        dist_list = []
        for j in range(len(model_hists)):
            V = model_hists[j]
            dist = dist_module.get_dist_by_name(Q, V, dist_type)
            D[j, i] = dist
        idx = list(D[:, i]).index(min(D[:, i]))
        best_match.append(idx)

    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist

    for img in image_list:

        # Gray or colored image
        if hist_isgray == True:
            img = rgb2gray(np.array(Image.open(img)))
        else:
            img = np.array(Image.open(img))

        # Histogram type
        hist = histogram_module.get_hist_by_name(np.array(img, float), num_bins, hist_type)

        # Save results
        image_hist.append(hist)

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

    plt.figure()

    num_nearest = 5  # show the top-5 neighbors

    best_match, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

    for i in range(len(query_images)):
        closest = sorted(range(len(D[:, i])), key=lambda k: D[:, i][k])[:num_nearest]
        for j in range(1, num_nearest+2):
            plt.figure(num_nearest+1, figsize=(20, 20))
            ax = plt.subplot(1, num_nearest+1, j)
            if j == 1:
                ax.title.set_size(25)
                ax.title.set_text('Q'+str(i))
                plt.axis('off')
                plt.sca(ax)
                plt.imshow(np.array(Image.open(query_images[i])))
            else:
                ax.title.set_size(25)
                ax.title.set_text('M'+str(round(D[closest[j-2], i], 4)))
                plt.axis('off')
                plt.sca(ax)
                plt.imshow(np.array(Image.open(model_images[closest[j-2]])))
        plt.show()
    return None

    
