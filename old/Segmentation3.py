import numpy as np
import pylab
import mahotas as mh
import skimage
from skimage.measure import regionprops 

dna = mh.imread('dna.jpeg') # convert from memory to disk array
dna = dna.squeeze() # change image array from 3d to 2d
pylab.gray() # convert to gray scale
pylab.imshow(dna)
#pylab.show() # print image

T = mh.thresholding.otsu(dna) # calculate a threshold value

# apply a gaussian filter that smoothen the image
dnaf = mh.gaussian_filter(dna, 8)
dnat = dnaf > T # do threshold

#labelling thereshold image
labeled, nr_objects = mh.label(dnat)
#print nr_objects # output number of objects
#pylab.imshow(labeled)
#pylab.jet() # makes image colourful

# find centroid
stats = skimage.measure.regionprops(labeled.astype(int))

#crop images
for x in range(nr_objects):
#for x in range(3):
	# find scale
	scale = np.maximum(int(stats[x].bbox[2]-stats[x].bbox[0]),int(stats[x].bbox[3]-stats[x].bbox[1]))+40
	print scale
	centroidValue=stats[x].centroid
	#print centroidValue
	y,x = labeled.shape
	startx = centroidValue[0]-scale/2
	starty = centroidValue[1]-scale/2
	if (centroidValue[0]<scale/2):
		startx = centroidValue[0]
	if (centroidValue[0]<scale/2):
		starty = centroidValue[0]
	pylab.imshow(dna[int(startx):int(startx+scale), int(starty):int(starty+scale)])
	pylab.show()
