from keras.models import load_model
import numpy as np 

import matplotlib.pyplot as plt


from skimage import io 
from skimage.color import rgb2gray
from skimage.transform import resize

model = load_model('model.h5')
print('Model Loaded')

def load_image(path,as_gray = False):
    return io.imread(path,plugin='matplotlib',as_gray=as_gray)

def get_gray_scale(image):
    return rgb2gray(image)

def set_scale(image,shape):
    return resize(image,shape)

def display(image):
    plt.imshow(image,cmap='gray')
    plt.show()

def get_image_for_network(path):
    img = load_image(path,True)
    img = set_scale(img,(224,224))
    img = img.reshape((224,224,1))
    img = img
    return img 

def get_prediction(path,original_shape):
    img = get_image_for_network(path)
    res = model.predict(np.asarray([img]))[0]
    return resize(res.reshape((224,224)),original_shape)


def predict_single(path):
    img = load_image(path)
    res = get_prediction(path,img.shape)
    display(res)

def predict_both(path):
    img = load_image(path)
    res = get_prediction(path,img.shape)
    maps = {'Real':img,'Predicted':res}
    plot_figures(maps,1,2)


def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    while True == True:
        path = input('Enter file path:')
        mode = input('Enter mode[1-single  2-double]:')
        if path == '':break
        if mode == '1':
            predict_single(path)
        elif mode == '2':
            predict_both(path)
        else:
            break
