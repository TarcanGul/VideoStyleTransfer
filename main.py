import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

# Config variables

SEED = 1789
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

tf.logging.set_verbocity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_VIDEO_PATH = ""
STYLE_IMG_PATH = ""

CONTENT_VIDEO_LEN = 5 #In seconds.

CONTENT_IMG_W = 500
CONTENT_IMG_H = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

NUM_FILTERS = 512 #Num of filters in block5_conv2

CONTENT_WEIGHT = 0.1
STYLE_WEIGHT = 1.0
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3

VGG19_MEAN_VALUES = [103.939, 116.779, 123.68]

'''
Helper functions
'''

'''
Deprocesses a matrix to an RGB image.
Implementation taken from https://keras.io/examples/neural_style_transfer/
'''
def deprocessImage(x):
    x = x.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    #These are imagenet mean values.
    x[:, :, 0] += VGG19_MEAN_VALUES[0]
    x[:, :, 1] += VGG19_MEAN_VALUES[1]
    x[:, :, 2] += VGG19_MEAN_VALUES[2]
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

'''
Loss Functions
'''


def styleLoss(style, gen):
    return STYLE_WEIGHT * (K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4. * (NUM_FILTERS**2) * ((STYLE_IMG_H * STYLE_IMG_W)**2)))


def contentLoss(content, gen):
    return CONTENT_WEIGHT * K.sum(K.square(gen - content))

#Calculating variation loss using the generated tensor.
#Implementation taken from https://keras.io/examples/neural_style_transfer/
#TODO: Change for video input.
def totalVariationLoss(x):
    a = K.square(
        x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, 1:, :CONTENT_IMG_W - 1, :])
    b = K.square(
        x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, :CONTENT_IMG_H - 1, 1:, :])
    return TOTAL_WEIGHT * K.sum(K.pow(a + b, 1.25))

'''
Pipeline functions
'''

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))

def preprocessData(data):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def styleTransfer(content_data, style_data, transfer_data):
    pass #TODO

'''
Main
'''

def main():
    print("Starting gif generator.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()