#main2.py
import os
import numpy as np
import pylab
import mahotas as mh
import glob     
import watershed 
import utils 
import plot
import csv
import main
from PIL import Image
import skimage
import skimage.io
import scipy
import pandas as pd
import click
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from IPython import embed
from tsne import bh_sne
import uuid

''' this part of the code was tested and it worked! commented out to save run time 

# Loads both nuclear (DAPI) and cytoplasm (SE) images
filenamesDAPI=list(glob.glob('./cell images/*ch1*.tiff'))
filenamesSE=list(glob.glob('./cell images/*ch2*.tiff'))

## Loops over each image in a directory
for imageNumber in range(len(filenamesDAPI)): 
    #print filenamesSE[imageNumber]
    #if not filenamesSE[imageNumber] == './cell images/r05c05f65p01-ch2sk1fk1fl1.tiff':
    #     continue
    
    # obtains image data
    DAPI = skimage.io.imread(filenamesDAPI[imageNumber]) 
    SE = skimage.io.imread(filenamesSE[imageNumber]) 
 
    # applys gaussina filter
    DAPIf = mh.gaussian_filter(DAPI, 13)
    SEf = mh.gaussian_filter(SE, 3)
    #pylab.imshow(DAPIf)
    #pylab.show()

    # finds threshold
    T = mh.thresholding.otsu(np.uint16(SEf))
    SEt= SEf > T/2
    #pylab.gray() 
    #pylab.imshow(DAPI)
    #pylab.show()

    # Finds seeds on the nuclear image
    rmax = mh.regmax(DAPIf)
    pylab.imshow(rmax)
    rmax[np.logical_not(SEt)] = 0
    #pylab.show()
    #pylab.imshow(mh.overlay(DAPI, rmax))# print image and rmax (with second channel in red)
    #pylab.show() # seeds only show when image is zoomed in

    # Watershed on the cytoplasm image
    seeds,nr_nuclei = mh.label(rmax) # nuclei count
    imagew = mh.cwatershed(-SEf, seeds)

    # Remove areas that aren't nuclei 
    SEn=np.logical_not(SEt)
    imagew[SEn] = 0
    #pylab.imshow(imagew.astype(np.int16)) # convert datatype from int64 to int16 to display labelled value in []
    #embed()
    #pylab.jet() 
    #pylab.show()

   # crop and label images
    
    utils.crop_and_save(SE, imagew,'./cropped/', filenames=None, square=True, resize=None, padding=5)
'''


# generating random cell ID for each cropped cells
cellID=[]
filenames=list(glob.glob('./cropped/*'))
for imageNumber in range(len(filenames)): 
    cellID.append(str(uuid.uuid4()))

# writing the cell IDs to a csv file
filename="file.csv" 
df = pd.DataFrame (cellID, columns=["UUID"]) # assign unique ID for each cell 
df.to_csv(filename)


cell_boundaries_list=[]
# loading image
filenames = list(glob.glob('./cropped/*'))
for idx,filename in enumerate(filenames):
    targetCell = skimage.io.imread(filenames[idx])
    dimension = targetCell.shape

    # calculating cell boundaries

    boolean_border = mh.bwperim(targetCell, n=4)
    cell_boundaries = np.ravel_multi_index(np.nonzero(boolean_border),dimension)

    # formatting the calculated cell boundaries
    CB_string=map(str, cell_boundaries) # converting all the elements in the array to string
    for x,y in enumerate(CB_string): # add ", " to all the elements so that when they are combined, they are still distinctly recognizable
        CB_string[x]=CB_string[x]+', ' 
    boundaries=''.join(CB_string) # combine all elements in an array to form a single string

    # adding the calculation to the cell_boundaries_list
    cell_boundaries_list.append(boundaries)

csv = "file.csv" 
df = utils.read_csv(csv)
print df.shape
df['cell boundary']=pd.Series (cell_boundaries_list)
df.to_csv(csv)

'''
# writing  cell boundaries to the csv file

csv = "file.csv" 
df = utils.read_csv(csv)
#print df.shape
df['cell boundaries'] = pd.Series (cell_boundaries_list)
df.to_csv(csv)'''



'''
csv = "file.csv" 
df = utils.read_csv(csv)
#CB = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
#pd.merge(df,{'cell boundaries':cell_boundaries_list })
df = pd.DataFrame(df,cell_boundaries_list,  columns=['A', 'cell boundaries'])
df.to_csv(csv)
'''



'''
notes from command window

bw=np.zeros((4,4)) // make a matrix 
//assign values
>>> bw[2,0]=1
>>> bw[2,1]=1
>>> bw[3,2]=1
>>> bw[1,3]=1

bw1=mahotas.bwperim(bw, n=4)//border true.false


np.nonzero(bw)

np.ravel_multi_index(np.nonzero(bw), (4,4))


np.ravel_multi_index(np.nonzero(bw1), (4,4))

'''



# 
# finding a list of pixels in cell bondaries



## (TODO) Loop over each cell found
  ## (TODO) Save cell to CSV with:
   # //1. a random UUID ()
   # //2. list of pixels in cell boundries 
    # single_cell_img = im == id
    # cell_perim = perim(im)  -----> mh.labeled.bwperim
    # locations = nonzero(cell_perim)
    # ravelled_locs = np.ravel_multi_index(locations, im.shape)
  # 3. filename of the image that the cell was found in











