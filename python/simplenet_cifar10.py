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

model_cifar10 = cifar10_simple_net(config, cnn_acc0)