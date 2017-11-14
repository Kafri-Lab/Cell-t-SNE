# utils.py
import numpy as np
import pylab
import mahotas as mh
import skimage
from IPython import embed
from skimage.measure import regionprops 
from matplotlib import pyplot as plt
import scipy.misc

import pandas as pd


def crop_and_save(image, labelled, save_location, filenames=None):
  nr_objects=int(np.max(labelled))
  stats = skimage.measure.regionprops(labelled.astype(int))

  for index in range(nr_objects):
    stat = stats[index]
    x_px_size = int(stat.bbox[2]-stat.bbox[0])
    y_px_size = int(stat.bbox[3]-stat.bbox[1])
    # find the largest resolution (x or y) of bounding box around cell
    resolution = np.maximum(x_px_size,y_px_size)
    resolution+=40 # add amount of pixels around cell
    centroidValue=stat.centroid
    y,x = labelled.shape
    startx = centroidValue[0]-resolution/2
    starty = centroidValue[1]-resolution/2
    if (centroidValue[0]<resolution/2):
      startx = centroidValue[0]
    if (centroidValue[0]<resolution/2):
      starty = centroidValue[0]

    result = image[int(startx):int(startx+resolution), int(starty):int(starty+resolution)]
    if result.size == 0:
      continue

    name = filenames[index] if filenames else index # Use a given filename for this crop or use the index when saving
    filename = '%s%s.jpg' % (save_location, name)
    print('Saving cropped cell to file: %s' % filename)
    scipy.misc.imsave(filename, result)
    

def read_csv(filename):
  df = pd.read_csv(filename)
  return df

def gray_to_color(img):
    if len(img.shape) == 2:
        img = np.invert(img)
        img = np.dstack((img, img, img))
    return img

def get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
