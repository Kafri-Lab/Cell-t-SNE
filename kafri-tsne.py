#!/anaconda/bin/python
# WORK THE PROGRESS
# TODO: implement the following functionality
#    python kafri-tsne.py --input-images barcode_images/ --perplexity 20 --output barcode_canvas.jpg

import click
from PIL import Image
from matplotlib import pyplot as plt
import glob
import numpy as np



@click.command()
@click.option('--input-folder', help='The folder that contains the images.')
@click.option('--perplexity', default=20, help='todo')
@click.option('--output', default='output.jpg', help='output to tsne image')
def tsne(input_folder, perplexity, output):
    """Simple program that greets NAME for a total of COUNT times."""
    images = []
    for filename in glob.glob('barcode_images/*.jpg'):
        print filename
        #import code
        #code.interact(local=locals())

        im=Image.open(filename)
        im=np.array(im)
        images.append(im)
        plt.imshow(im)
        raw_input("press enter..")




if __name__ == '__main__':
    plt.ion()
    tsne()



