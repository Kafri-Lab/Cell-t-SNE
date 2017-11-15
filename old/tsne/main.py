# main.py
import numpy as np
import pylab
import mahotas as mh
import glob     
import watershed # label image by calling watershed.py
import utils # crop cell by calling utils.py
import tsne1
from PIL import Image
import skimage
import skimage.io
import scipy
'''
# crop and label images
for filename in list(glob.glob('*.tif')): 
	img = skimage.io.imread(filename) # read and save image as an array (img.shape is (1024, 1360))
	y_value = watershed.label(img) # watershed images; y_value stores the labels
	utils.crop (img, y_value) #crop image
'''
#put together a list of x value for tsne to process
filenames=list(glob.glob('*.jpg'))
x_value = np.zeros((4900, len(filenames))) # Dimension of the image: 70*70=4900; x_value will store images in 2d array
for imageName in filenames : 
	count = 0
	image1d = scipy.misc.imresize(skimage.io.imread(imageName), (70,70)) #reshape size to 70,70 for every image
	image1d = image1d.flatten() #image1d stores a 1d array for each image
	x_value[:,count] = image1d # add a row of values
	count += 1

tsne1.tSNE(x_value) #plot data using tsne