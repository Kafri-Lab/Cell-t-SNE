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
@click.option('--channel', help='The image channel to crop.', default=1)
@click.option('--square/--rectangle', help='Crop the cell in into a square box rather than a rectangle.', default=False)
def crop_images(csv,channel,square):
  df = utils.read_csv(csv)
  print('Found number of cells: %s' % df.shape[0])
  image_dir = './images/Ron/'
  # Loop by image
  for image_filename in df.FileName.unique():
    cells_in_img = df.loc[df['FileName'] == image_filename]
    cell_ids = []
    # Set the channel number in the image filename to load
    s = list(image_filename)
    s[15]=str(channel)
    ch_image_filename = "".join(s)
    # Load image
    image = skimage.io.imread(image_dir + ch_image_filename)

    # Build labelled image
    labelled = np.zeros(image.shape)
    count = 1
    for row_index, row in cells_in_img.iterrows():
      cyto_px_ids = map(int, row.cyto_boundaries.split()) # get locations of this cell's boundries as a list of ints
      labelled[np.unravel_index(cyto_px_ids, labelled.shape, order='F')] = count # set this cell in the labelled image
      cell_ids.append(row.CellID)
      count+=1

    save_location = './images/cropped_images/Ron/ch%s-' % channel
    utils.crop_and_save(image, labelled, save_location, filenames=cell_ids, square=square)
  print('Saved number of cropped cells: %s' % df.shape[0])

@click.command()
@click.option('--csv', help='The csv file that contains single cell data.', required=True)
@click.option('--color-by', help='The measurement name to color the boxes by.', default='Trace')
@click.option('-x', help='The measurement name on the X axis.', required=True)
@click.option('-y', help='The measurement name on the Y axis.', required=True)
@click.option('--dpi', help='The resolution to save the output image.', default=200)
@click.option('--channel', help='The image channel to display.', default=1)
def image_scatter(csv,color_by,x,y,dpi,channel):

  df = utils.read_csv(csv)
  print('Found number of cells: %s' % df.shape[0])
  image_dir = './images/cropped_images/Ron/'
  cell_imgs = []
  colors = []
  xx = np.array([])
  yy = np.array([])

  cmap='gist_rainbow'
  color_list = utils.get_colors(len(np.unique(df[color_by])),cmap=cmap)

  # blue is better when there are only 3 colours
  if len(color_list) == 3 and cmap == 'gist_rainbow':
    color_list[2] = (0.05, 0.529, 1, 1.0)

  for row_id, row in df.iterrows():
    cell_id = row.CellID
    image_filename = image_dir + 'ch' + str(channel) + '-' + cell_id + '.jpg'
    if not os.path.exists(image_filename):
      print('[WARN] no image found %s'% image_filename)
      continue
    print('Loading image %s' % image_filename)
    image = skimage.io.imread(image_filename)
    image = utils.gray_to_color(image)
    cell_imgs.append(image)
    xx = np.append(xx,row[x])
    yy = np.append(yy,row[y])
    color_id = np.where(np.unique(df[color_by])==row[color_by])[0][0] # find position where this value appears
    c = color_list[color_id]
    c = (int(c[0]*255),int(c[1]*255),int(c[2]*255)) # convert value range, ex. 1.0 -> 255 or 0.0 -> 0
    colors.append(c)

  if len(cell_imgs)==0:
    print('[ERROR] 0 cropped single cell images found.')
    return

  canvas = plot.image_scatter(yy, xx, cell_imgs, colors, min_canvas_size=4000)
  plt.imshow(canvas,origin='lower')
  plt.title('%s vs %s' % (x,y))
  plt.xlabel('%s' % x)
  plt.ylabel('%s' % y)
  plt.xticks([])
  plt.yticks([])
  patches=[]
  for i in range(len(np.unique(df[color_by]))):
    label = '%s %s' % (color_by, np.unique(df[color_by])[i])
    if color_by == 'Dend.cat':
      label = 'Detected Category %s' % (i+1)
    # Plot additional data that can't be in the main csv
    # extra_datafile = 'PCaxes.csv'
    # if os.path.exists(extra_datafile):
    #   df_extra = utils.read_csv(extra_datafile)
    #   from IPython import embed
    #   embed() # drop into an IPython session
    #   plt.scatter(avg_x, avg_y,c=color,marker='*')
    patch = mpatches.Patch(color=color_list[i], label=label)
    patches.append(patch)

  plt.legend(handles=patches,bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
  save_location = './images/%s_image_scatter_by_%s_dpi%s.jpg' % (csv, color_by, dpi)
  plt.savefig(save_location,dpi=dpi,pad_inches=1,bbox_inches='tight')
  # plt.show()
  print('Saved image scatter to %s' % save_location)

# Setup group of command line commands
@click.group()
def cli():
    pass
cli.add_command(crop_images)
cli.add_command(image_scatter)

if __name__ == '__main__':
  cli()  # make command line commands available
  # everything()


# import pylab
# pylab.imshow(labelled);pylab.show()