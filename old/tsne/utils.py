# utils.py
import numpy as np
import pylab
import mahotas as mh
import skimage
from IPython import embed
from skimage.measure import regionprops 
import scipy.misc

def crop(image, labelled):
	nr_objects=np.max(labelled)
	# find centroid
	stats = skimage.measure.regionprops(labelled.astype(int))
	#crop images
	#croppedImg= ""
	for cellID in range(nr_objects):
		# find scale
		scale = np.maximum(int(stats[cellID].bbox[2]-stats[cellID].bbox[0]),int(stats[cellID].bbox[3]-stats[cellID].bbox[1]))+40
		centroidValue=stats[cellID].centroid
		y,x = labelled.shape
		startx = centroidValue[0]-scale/2
		starty = centroidValue[1]-scale/2
		if (centroidValue[0]<scale/2):
			startx = centroidValue[0]
		if (centroidValue[0]<scale/2):
			starty = centroidValue[0]

		result = image[int(startx):int(startx+scale), int(starty):int(starty+scale)]
		if result.size == 0:
			continue
		scipy.misc.imsave('croppedCell%s.jpg' % cellID, result)
		