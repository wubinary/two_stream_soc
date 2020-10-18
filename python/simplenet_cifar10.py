import cv2, time, configparser
import ctypes
import numpy as np

from pynq import Overlay
from pynq import Xlnk

from VGG16.accelerator import CNN_accelerator
from VGG16.vgg import cifar10_simple_net

config_path = './files/config.config'
config = configparser.ConfigParser()   
config.read(config_path)

overlay = Overlay(config["FPGAConfig"]["bitstream_path"])

(in_height, in_width, in_channel) = \
        (int(config["DataConfig"]["image_height"]),int(config["DataConfig"]["image_width"]),3)

cnn_acc0 = CNN_accelerator(config, overlay.DoCompute_0)

cifar10_model = cifar10_simple_net(config, cnn_acc0)

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def numpy_quantize_tensor_scale_zeropoint(x, num_bits=8, scale=None, zeropoint=None):
    
    qmin = 0.
    qmax = 2.**num_bits - 1.
    #scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zeropoint + x / scale
    q_x = np.round(np.clip(q_x,qmin,qmax))
    
    #q_x.clamp_(qmin, qmax).round_()
    #q_x = q_x.round().byte()
    return np.array(q_x).astype(np.uint8)

def normalization(img):
    img_nor = img/255.
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean = np.array(mean).reshape(1,1,3)
    std = np.array(std).reshape(1,1,3)
    return (img_nor-mean)/std


def inference(img):
    x = normalization(img)
    x = numpy_quantize_tensor_scale_zeropoint(x,8,0.11948869665231317,7)
    
    output = cifar10_model(x)
#     print(output)
    return cifar10_classes[np.argmax(output)]