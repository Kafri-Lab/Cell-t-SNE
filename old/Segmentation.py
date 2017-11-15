import numpy as np
import pylab
import mahotas as mh

dna = mh.imread('dna.jpeg') # convert from memory to disk array
dna = dna.squeeze() # change image array from 3d to 2d
pylab.gray() # convert to gray scale
pylab.imshow(dna)
pylab.show() # print image

#print dna.dtype
#print dna.max()
#print dna.min()

pylab.imshow(dna // 2) # doesn't change image

#prints a blocky black and white image
T = mh.thresholding.otsu(dna) # generate an int value (of 45)
pylab.imshow(dna > T)
pylab.show()

# apply a gaussian filter that smoothen the image
dnaf = mh.gaussian_filter(dna, 8)
dnat= dnaf > T
pylab.gray()
nuclei1= dnat
pylab.imshow(dnat)
pylab.show()

#labelling thereshold image
labeled, nr_objects = mh.label(dnat)
print nr_objects # output number of objects
pylab.imshow(labeled)
pylab.jet() # makes image colourful
pylab.show() 

dnaf = mh.gaussian_filter(dnaf, 8)
rmax = mh.regmax(dnaf)
pylab.imshow(mh.overlay(dna, rmax))# print dna and rmax (with second channel in red)
pylab.show() # seeds only show when image is zoomed in

dnaf = mh.gaussian_filter(dnaf, 16)# apply different filter to yield better result
rmax = mh.regmax(dnaf)
pylab.imshow(mh.overlay(dna, rmax))
pylab.show()

seeds,nr_nuclei = mh.label(rmax) # nuclei count
print nr_nuclei # unlike the example, the result is 36 compared to 22

dist = mh.distance(dnat)
dist = dist.max() - dist
dist -= dist.min()
dist = dist/float(dist.ptp()) * 255
dist = dist.astype(np.uint8)
pylab.imshow(dist)
pylab.show()

nuclei = mh.cwatershed(dist, seeds)
pylab.imshow(nuclei)
pylab.show()

pylab.show()
nuclei[nuclei1] = 0
pylab.imshow(nuclei1)
pylab.show()
'''

images = map(Image.open, ['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('test.jpg')
'''

'''
borders = np.zeros(nuclei.shape, np.bool)
borders[ 0,:] = 1
borders[-1,:] = 1
borders[:, 0] = 1
borders[:,-1] = 1
at_border = np.unique(nuclei[borders])
for obj in at_border:
    whole[whole == obj] = 0
pylab.imshow(whole)
pylab.show()
'''

