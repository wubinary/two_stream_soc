import cv2, time, configparser
import ctypes
import numpy as np

from pynq import Overlay
from pynq import Xlnk

import VGG16.fpga_nn as fpga_nn
from VGG16.accelerator import CNN_accelerator


def make_layers(config, in_channel=3, accelerator=None):
    assert config is not None    
    in_height = int(config["DataConfig"]["image_height"])
    in_width = int(config["DataConfig"]["image_width"])
    in_channel = in_channel

    assert accelerator is not None
    acc = accelerator

    layers = []
    #Conv(output channel, input channel, input height, input width, kerSize, stride)
    layers += [fpga_nn.Conv2DPool(32, in_channel, in_height, in_width, ker = 3, poolWin = 2, accelerator=acc)]

    layers += [fpga_nn.Conv2DPool(64, 32, int(in_height/2), int(in_width/2), ker = 3, poolWin = 2, accelerator=acc)]

    layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/4), int(in_width/4), ker = 3, poolWin = 2, accelerator=acc)]
    
    layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/8), int(in_width/8), ker = 3, poolWin = 2, accelerator=acc)]

    layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/16), int(in_width/16), ker = 3, poolWin = 2, accelerator=acc)]

#     layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/32), int(in_width/32), ker = 3, poolWin = 2, accelerator=acc)]

    # conv output size = (8,8,512)
    layers += [fpga_nn.Flatten(int(in_height/32), int(in_width/32), 64)]
    layers += [fpga_nn.Linear(512,int(in_height/32)*int(in_width/32)*64)]
    layers += [fpga_nn.Linear(101,512, quantize = False)]

    return layers

class UCF101VGG7(CNN_accelerator):
    def __init__(self, config, layers, params_path = None):
        super(UCF101VGG7, self).__init__(config, is_spatial=True)
        self.layers = layers
        self.params_path = params_path

        # initialize weight for each layer
        self.init_weight(params_path)

        # copy weight data to hardware buffer 
        self.load_parameters();

def ucf101_vgg7(model_path, config, accelerator):
    layers = make_layers(config, in_channel=3, accelerator=accelerator)
    params_path = model_path
    model = UCF101VGG7(config, layers, params_path = params_path)
    return model        
        
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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape(1,1,3)
    std = np.array(std).reshape(1,1,3)
    return (img_nor-mean)/std


def inference(img):
    x = normalization(img)
    x = numpy_quantize_tensor_scale_zeropoint(x,num_bits=8, \
        scale=0.3036378016349125, zeropoint=6)
    
    output = cifar10_model(x)
#     print(output)
    return cifar10_classes[np.argmax(output)]

if __name__=='__main__':
    config_path = './files/config.config'
    model_path = './files/params/ucf101_vgg7/model.pickle'
    config = configparser.ConfigParser()   
    config.read(config_path)

    overlay = Overlay(config["FPGAConfig"]["bitstream_path"])

    (in_height, in_width, in_channel) = \
            (int(config["DataConfig"]["image_height"]),int(config["DataConfig"]["image_width"]),3)

    ucf101_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #########################################################################
    cnn_acc0 = CNN_accelerator(config, overlay.DoCompute_0)

    ucf101_spatial_model = ucf101_vgg7(model_path, config, cnn_acc0)

    #########################################################################