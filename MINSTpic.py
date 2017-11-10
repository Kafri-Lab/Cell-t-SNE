#!/Users/wunina/anaconda2/bin/python
#from IPython.core.debugger import Tracer; breakpoint = Tracer()
from tsne import bh_sne
import numpy as np
from skimage.transform import resize
import time
from matplotlib import pyplot as plt

def d_to_3d(img): 
    if len(img.shape) == 2: # do nothing if already more than 2 dimensions
        img = np.invert(img)
        img = np.dstack((img, img, img)) # add a third dimension
    return img

def min_resize(img, size):
    """
    Resize an image so that it is exactly the given size along the minimum spatial dimension. 
    Also, scale the other dimension porportionally if nessecary.
    """
    w, h = float(img.shape[0]), float(img.shape[1])
    if min([w, h]) != size:
        if w <= h:
            img = resize(img, (int(round((h/w)*size)), int(size))) # 
        else:
            img = resize(img, (int(size), int(round((w/h)*size))))
    return img

def image_scatter(images, img_res, res=4000, cval=1.):
    """
    Embeds images via tsne into a scatter plot.
    Parameters
    ---------
    features: numpy array
        Features to visualize
    images: list or numpy array
        Corresponding images to features. Expects float images with values from (0,1).
    img_res: float or int
        Resolution to embed images at into the canvas
    res: float or int  # TODO: change variable name from res to min_canvas_res
        Size of canvas, the minimum size of either x or y
    cval: float or numpy array
        Background color value
    Returns
    ------
    canvas: numpy array
        Image of visualization
    """

    #from IPython import embed
    #embed()
    features = np.copy(features).astype('float64')
    images = [d_to_3d(image) for image in images] # TODO: rename funnnnction from gray_to_color to 2d_to_3d
    images = [min_resize(image, img_res) for image in images]
    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])

    f2d = bh_sne(features) # docs: https://github.com/danielfrg/tsne
    # alternative: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE

    # Get locations of each image on the t-sne plot
    xx = f2d[:, 0] # x values for each image
    yy = f2d[:, 1] # y values for each image

    # Get scatter plot axis limits (min and max)
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()

    # Calculate canvas size
    scale_canvas_x = (x_max-x_min)
    scale_canvas_y = (y_max-y_min)
    if scale_canvas_x > scale_canvas_y:
        canvas_res_x = scale_canvas_x/float(scale_canvas_y)*res
        canvas_res_y = res
    else:
        canvas_res_x = res
        canvas_res_y = scale_canvas_y/float(scale_canvas_x)*res

    # Create canvas by embedding images at the correct positions according it t-sne
    canvas = np.ones((int(canvas_res_x)+max_width, int(canvas_res_y)+max_height, 3))*cval
    # TODO: Replace this to use a scale factor
    x_coords = np.linspace(x_min, x_max, canvas_res_x) # TODO: Replace this to use a scale factor
    y_coords = np.linspace(y_min, y_max, canvas_res_y) # TODO: Replace this to use a scale factor
    for x, y, image in zip(xx, yy, images):
        w, h = image.shape[:2]
        scaled_x = np.argmin((x - x_coords)**2) # TODO: Replace this to use a scale factor
        scaled_y = np.argmin((y - y_coords)**2) # TODO: Replace this to use a scale factor
        canvas[scaled_x:scaled_x+w, scaled_y:scaled_y+h] = image # embed image
    return canvas

def main():
    from skdata.mnist.views import OfficialImageClassification
    # load up data
    data = OfficialImageClassification(x_dtype="float32")
    x_data = data.all_images
    y_data = data.all_labels

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(x_data).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1)) # vectorize each image

    # For speed of computation, only run on a subset
    n = 300
    x_data = x_data[:n]
    y_data = y_data[:n]

    image_size=28
    plot_size=800
    canvas = image_scatter(x_data, data.all_images, image_size, plot_size)
    plt.ion() # maybe not needed? for actually seeing what is imshow'ed
    plt.imshow(canvas)
    plt.show(block=False) # actually show the image

    raw_input ("Press Enter..") # wait before closing image so human sees image

      #import code
    #code.interact(local=dict(globals(), **with values locals()))


    #plt.close("all") # not needed?
      # TODO: change variable name from res to canvas_res
    #plt.imsave('canvas.jpg', canvas) # TODO: Use variable output name instead of canvas.jpg
