import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    
    zipped = list(map(min, list(zip(x,y))))
    return 1 - (0.5*((np.sum(zipped)/np.sum(x)) + (np.sum(zipped)/np.sum(y))))


# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    
    dist = np.sum((x - y)**2)
    assert 0 <= dist <= np.sqrt(2), "not in [0, sqrt(2)]"
    return dist


# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    
    x += 0.5
    y += 0.5
    dist = np.sum(((x - y)**2)/((x) + (y)))
    assert 0 <= dist <= np.inf, "not in [0, inf]"
    return dist



def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
  




