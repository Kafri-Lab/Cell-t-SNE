#!/usr/bin/env python
import os
import numpy as np
import pylab
import mahotas as mh
import glob     
import watershed # label image by calling watershed.py
import utils # crop cell by calling utils.py
import plot
# import tsne1
from PIL import Image
import skimage
import skimage.io
import scipy
import pandas as pd
import click
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from tsne import bh_sne
import code
from IPython import embed



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
def crop_images(csv):
  df = utils.read_csv(csv)
  print('Found number of cells: %s' % df.shape[0])
  image_dir = './images/LFS images/'
  for image_filename in df.FileName.unique():
    image = skimage.io.imread(image_dir+image_filename)
    labelled = np.zeros(image.shape)
    cells_in_img = df.loc[df['FileName'] == image_filename]
    cell_ids = []

    count = 1
    for row_index, row in cells_in_img.iterrows():
      cyto_px_ids = map(int, row.cyto_boundaries.split()) # get locations of this cell's boundries as a list of ints
      labelled[np.unravel_index(cyto_px_ids, labelled.shape, order='F')] = count # set this cell in the labelled image
      cell_ids.append(row.CellID)
      count+=1


    save_location = './images/cropped_images/'
    utils.crop_and_save(image, labelled, save_location, filenames=cell_ids)
  print('Saved number of cropped cells: %s' % df.shape[0])
  #     save the location of the cropped image into csv


@click.command()
@click.option('--csv', help='The csv file that contains single cell data.', required=True)
def tsne_images(csv):
  image_dir = './images/cropped_images/'
  filenames=list(glob.glob(image_dir+'*.jpg'))
  '''
  x_value = np.zeros((4900, len(filenames))) # Dimension of the image: 70*70=4900; x_value will store images in 2d array
  for imageName in filenames: 
    count = 0
    image1d = scipy.misc.imresize(skimage.io.imread(imageName), (70,70)) #reshape size to 70,70 for every image
    image1d = image1d.flatten() #image1d stores a 1d array for each image
    x_value[:,count] = image1d # add a row of values
    count += 1
  '''
  x_value = np.zeros((len(filenames),4900)) # Dimension of the image: 70*70=4900; x_value will store images in 2d array
  print filenames
  count = 0
  for imageName in filenames: 
    image1d = scipy.misc.imresize(skimage.io.imread(imageName), (70,70)) #reshape size to 70,70 for every image
    image1d = image1d.flatten() #image1d stores a 1d array for each image
    x_value[count,:] = image1d # add a row of values
    #embed()
    count += 1
    if count>50:
      break

  print x_value.shape
  vis_data = bh_sne(x_value,perplexity=5)# tsne embedding
  print vis_data.shape
  vis_x = vis_data[:, 0]
  vis_y = vis_data[:, 1]
  
 
  df = utils.read_csv(csv)
  print df.shape
  df['tsne1']=pd.Series (vis_x)
  df['tsne2']=pd.Series (vis_y)
  df.to_csv(csv)
  
@click.command()
@click.option('--csv', help='The csv file that contains single cell data.', required=True)
@click.option('--colour', help='The measurement name to colour the boxes by.', default='Trace')
@click.option('-x', help='The measurement name on the X axis.', required=True)
@click.option('-y', help='The measurement name on the Y axis.', required=True)
@click.option('--dpi', help='The resolution to save the output image.', default=200)

def image_scatter(csv,colour,x,y,dpi):
  df = utils.read_csv(csv)
  print('Found number of cells: %s' % df.shape[0])
  image_dir = './images/cropped_images/'
  cell_imgs = []
  colours = []
  xx = np.array([])
  yy = np.array([])

  color_map = utils.get_cmap(len(np.unique(df[colour])))

  for row_id, row in df.iterrows():
    cell_id = row.CellID
    image_filename = image_dir + cell_id + '.jpg'
    if not os.path.exists(image_filename):
      print('[WARN] no image found %s'% image_filename)
      continue
    print('Loading image %s' % image_filename)
    image = skimage.io.imread(image_filename)
    image = utils.gray_to_color(image)
    cell_imgs.append(image)
    xx = np.append(xx,row[x])
    yy = np.append(yy,row[y])
    color_id = np.where(np.unique(df[colour])==row[colour])[0][0] # find position where this value appears
    color = color_map(color_id)
    color = (int(color[0]*255),int(color[1]*255),int(color[2]*255))
    colours.append(color)
  canvas = plot.image_scatter(xx, yy, cell_imgs, colours, min_canvas_size=4000)

  plt.imshow(canvas)
  plt.title('%s vs %s' % (x,y))
  plt.xlabel('%s' % x)
  plt.ylabel('%s' % y)
  patches=[]
  for i in range(len(np.unique(df[colour]))):
    # print i
    # print color_map(i)
    patch = mpatches.Patch(color=color_map(i), label='%s %s' % (colour, np.unique(df[colour])[i]))
    patches.append(patch)
  plt.legend(handles=patches,fontsize=10)
  plt.legend(handles=patches,bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
  # plt.show()

  save_location = './images/%s_image_scatter_by_%s_dpi%s.jpg' % (csv, colour, dpi)
  plt.savefig(save_location,dpi=dpi)
  # plt.savefig('image.jpg',dpi=1200)
  # scipy.misc.imsave(save_location, canvas)
  print('Saved image scatter to %s' % save_location)

# Setup group of command line commands
@click.group()
def cli():
    pass
cli.add_command(crop_images)
cli.add_command(image_scatter)
cli.add_command(tsne_images)

if __name__ == '__main__':
  cli()  # make command line commands available
  # everything()


# import pylab
# pylab.imshow(labelled);pylab.show()