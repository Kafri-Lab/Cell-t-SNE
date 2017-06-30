#!/Users/wunina/anaconda2/bin/python
# WORK THE PROGRESS
# TODO: implement the following functionality
#    python kafri-tsne.py --input-images barcode_images/ --perplexity 20 --output barcode_canvas.jpg

import click
from PIL import Image
from matplotlib import pyplot as plt
import glob
import numpy as np
from skdata.mnist.views import OfficialImageClassification
from tsne import bh_sne


@click.command()
@click.option('--input-folder', help='The folder that contains the images.')
@click.option('--perplexity', default=20, help='todo')
@click.option('--output', default='output.jpg', help='output to tsne image')
def tsne(input_folder, perplexity, output):
    
    data = OfficialImageClassification(x_dtype="float32")

    x_data = data.all_images
    y_data = data.all_labels
    import code
    #code.interact(local=locals())


    images = []

# convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(x_data).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

# For speed of computation, only run on a subset
    n = 300
    x_data = x_data[:n]
    y_data = y_data[:n]

# perform t-SNE embedding
    vis_data = bh_sne(x_data)

# plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    cmap = plt.cm.get_cmap("jet", 10)
    plt.scatter(vis_x, vis_y, c=y_data)
# plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)
    plt.show()

    filenames= glob.glob('barcode_images/*.jpg')
    for filename in filenames:
        print filename

        im=Image.open(filenames[0])
        im=np.array(im)

        images.append(im)
        plt.imshow(im)
        code.interact(local=dict(globals(), **locals()))
        plt.imshow(np.squeeze(x_data[0]))
        raw_input("press enter..")

if __name__ == '__main__':
    plt.ion()
    tsne()
