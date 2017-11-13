#!/usr/bin/env python
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
import pandas as pd
import click


def everything():
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

@click.command()
@click.option('--csv', help='The csv file that contains single cell data.', required=True)
def create_cropped_images_given_csv(csv):
  df = utils.read_csv(csv)
  print('Found number of cells: %s' % df.shape[0])
  image_dir = './images/Ron/'
  for image_filename in df.FileName.unique():
    image = skimage.io.imread(image_dir + image_filename)
    labelled = np.zeros(image.shape)
    cells_in_img = df.loc[df['FileName'] == image_filename]
    cell_ids = []

    count = 1
    for row_index, row in cells_in_img.iterrows():
      cyto_px_ids = map(int, row.cyto_boundaries.split()) # get locations of this cell's boundries as a list of ints
      labelled[np.unravel_index(cyto_px_ids, labelled.shape, order='F')] = count # set this cell in the labelled image
      cell_ids.append(row.CellID)
      count+=1


    save_location = './images/cropped_images/Ron/'
    utils.crop_and_save(image, labelled, save_location, filenames=cell_ids)
  print('Saved number of cropped cells: %s' % df.shape[0])
  #     save the location of the cropped image into csv


if __name__ == '__main__':
  # everything()
  create_cropped_images_given_csv()


# import pylab
# pylab.imshow(labelled);pylab.show()