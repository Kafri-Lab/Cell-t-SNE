import numpy as np
import pylab
import mahotas as mh

## (TODO) Loop over each image in a directory
  ## (TODO) Load both nuclear (DAPI) and cytoplasm (SE) images

  ## (TODO) Find seeds on the nuclear image
  ## (TODO) Watershed on the cytoplasm image

dna = mh.imread('dna.jpeg') # convert from memory to disk array
dna = dna.squeeze() # change image array from 3d to 2d
pylab.gray() # convert to gray scale
pylab.imshow(dna)
#pylab.show() # print image

T = mh.thresholding.otsu(dna) # calculate a threshold value

# apply a gaussian filter that smoothen the image
dnaf = mh.gaussian_filter(dna, 8)
dnat= dnaf > T # do threshold

#labelling thereshold image
labeled, nr_objects = mh.label(dnat)
#print nr_objects # output number of objects
#pylab.imshow(labeled)
#pylab.jet() # makes image colourful

# Watershed
dnaf = mh.gaussian_filter(dna, 28)
rmax = mh.regmax(dnaf)
pylab.imshow(mh.overlay(dna, rmax))# print dna and rmax (with second channel in red)
#pylab.show() # seeds only show when image is zoomed in
dist = mh.distance(dnat)
seeds,nr_nuclei = mh.label(rmax) # nuclei count
dnaw = mh.cwatershed(dist, seeds)
pylab.imshow(dnaw)
#pylab.show()

# Remove areas that aren't nuclei (ie. that are value BLAH in the thresholded image (dnat))
dnat=np.logical_not(dnat)
dnaw[dnat] = 0

pylab.imshow(dnaw)
pylab.jet() # makes image colourful
pylab.show()

## (TODO) Loop over each cell found
  ## (TODO) Save cell to CSV with:
   # 1. a random UUID ()
   # 2. list of pixels in cell boundries 
    # single_cell_img = im == id
    # cell_perim = perim(im)
    # locations = nonzero(cell_perim)
    # ravelled_locs = np.ravel_multi_index(locations, im.shape)
  # 3. filename of the image that the cell was found in





