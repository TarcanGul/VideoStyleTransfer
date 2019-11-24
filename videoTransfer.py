
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
from imutils.video import FPS
import imutils
import cv2
from imutils.video import FileVideoStream

# Need to install FileVideostream and opencv in order to recieve the videstream and compute each of the frame

random.seed(1618)
np.random.seed(1618)
tf.compat.v1.set_random_seed(1618)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "inputVideo.mp4"           #TODO: Add this.
STYLE_IMG_PATH = "styleImg.jpg"             #TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 1


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (500,500))


#=============================<Helper Fuctions>=================================

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    print(img.shape)
    img = img.reshape(( CONTENT_IMG_H, CONTENT_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB' 으로 변환합니다.
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img
    # TODO: return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def eval_loss_and_grads(x):
    x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    # return None   #TODO: implement.
    S = gramMatrix(style)
    C = gramMatrix(gen)
    channels = 3
    size = STYLE_IMG_H * STYLE_IMG_W
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
   

def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    # return None   #TODO: implement.
    a = K.square(
        x[:, :STYLE_IMG_H - 1, :STYLE_IMG_W - 1, :] - x[:, 1:, :STYLE_IMG_W - 1, :])
    b = K.square(
        x[:, :STYLE_IMG_H - 1, :STYLE_IMG_W - 1, :] - x[:, :STYLE_IMG_W - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))





#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content video URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    
    cap = cv2.VideoCapture(CONTENT_IMG_PATH)
    content_frames = [] #For content data
    transfer_frames = [] #For transfer data
    imageIsRead, frame = cap.read()
    while imageIsRead:
        frame = imutils.resize(frame, width=500)
        tImg = frame.copy()
        content_frames.append(frame)
        transfer_frames.append(tImg)   
        imageIsRead, frame = cap.read()

    sImg = load_img(STYLE_IMG_PATH)
    print("Images have been loaded.")
    return ((content_frames, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (transfer_frames, CONTENT_IMG_H, CONTENT_IMG_W))

    #cap.release()
    #cv2.destroyAllWindows()
    #out.release()
    # tImg = cImg.copy()
    # sImg = load_img(STYLE_IMG_PATH)
    # print("      Images have been loaded.")
    # return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))

def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def preprocessDataBuffer(raw):
    frames, ih, iw = raw
    for i in range(len(frames)):
        frames[i] = img_to_array(frames[i])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames[i] = imresize(frames[i] , (ih, iw, 3))
    
        frames[i]  = frames[i] .astype("float64")
        frames[i]  = np.expand_dims(frames[i] , axis=0)
        frames[i]  = vgg19.preprocess_input(frames[i])
    return frames


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    print("Number of frames: ", len(cData))
    print("cData frame shape:" , cData[0].shape)
    print("sData shape:" , sData.shape)
    print("tData shape: ", tData[0].shape)
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    
    contentTensor = K.variable(cData[0])
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet",input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    frames_len = len(cData) #cData should contain individual frames
    output_frames = []
    for i in range(frames_len):
        print("Frame " + str(i))
        current_frame = cData[i]
        loss = 0.0
        loss += CONTENT_WEIGHT * contentLoss(contentOutput,genOutput)
        for layerName in styleLayerNames:
            # loss += None   #TODO: implement.
            layer_features = outputDict[layerName]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = styleLoss(style_reference_features, combination_features)
            loss += (STYLE_WEIGHT / len(styleLayerNames)) * sl        

        loss += TOTAL_WEIGHT * totalLoss(genTensor)
        # TODO: Setup gradients or use K.gradients().
        grads = K.gradients(loss, genTensor)
        outputs = [loss]
        outputs += grads
        global f_outputs
        f_outputs = K.function([genTensor], outputs)
    
        for _ in range(TRANSFER_ROUNDS):
            print("   Step %d." % i)
            #TODO: perform gradient descent using fmin_l_bfgs_b.
            current_frame, tLoss, info = fmin_l_bfgs_b(evaluator.loss, current_frame.flatten(),
                                        fprime=evaluator.grads, maxfun=10, maxiter=10, iprint=1)
            print("      Loss: %f." % tLoss)
            img = deprocessImage(current_frame.copy())
            print(img.shape)
            # saveFile = "finalOut%d.png" % i  #TODO: Implement.
            # imsave(saveFile, img)   #Uncomment when everything is working right.
            output_frames.append(img)

    for frame in output_frames:
        out.write(frame)
    out.release()
    print("   Transfer complete.")





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessDataBuffer(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessDataBuffer(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
