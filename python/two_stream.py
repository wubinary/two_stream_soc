import cv2, time, configparser
import ctypes
import numpy as np
import multiprocessing as mp
from multiprocessing import Process

from pynq import Overlay
from pynq import Xlnk
from LK_optical_flow.accelerator import LK_accelerator
from LK_optical_flow.utils import showarray, Feature_bank

from VGG16.accelerator import CNN_accelerator
from VGG16.vgg import simple_net, simple_net_2

class Two_stream(object):
    def __init__(self, config_path = './files/config.config'):
        
        config = configparser.ConfigParser()   
        config.read(config_path)
                
        overlay = Overlay(config["FPGAConfig"]["bitstream_path"])
        
        (h, w, c) = (self.in_height, self.in_width, self.in_channel) = \
                    (int(config["DataConfig"]["image_height"]),
                     int(config["DataConfig"]["image_width"]),
                     3)
        
        self.lucas_kanade_acc = LK_accelerator(config, overlay=overlay)
        self.feature_bank = Feature_bank(config)
        
        cnn_acc0 = CNN_accelerator(config, overlay.DoCompute_0)
        cnn_acc1 = CNN_accelerator(config, overlay.DoCompute_1)
        
        self.model_spatial = simple_net(config, cnn_acc0)
        self.model_temporal = simple_net_2(config, cnn_acc1)
        
        self.output = mp.Array(ctypes.c_uint, 101*2)# {'spatial':mp.Array('i', 101), 'temporal':mp.Array('i', 101)}
                
    def spatial_job(self, input_frame, output):
        out = self.model_spatial(input_frame)
        self.output[:101] = out
    
    def temporal_job(self, input_frame, old_frame, output):
        #input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        #old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        
        #vx,vy = self.lucas_kanade_acc.compute(old_frame, input_frame)
        
        #self.feature_bank.push(vx,vy)
        
        temp_frame = self.feature_bank.get_np_arr()
        
        out = self.model_temporal(temp_frame[:,:,:16]) ########################################## TODO: #####################
        self.output[101:] = out
            
    def __call__(self, input_frame, old_frame) -> np.ndarray :
        assert isinstance(input_frame, np.ndarray) and input_frame.dtype == np.uint8
        assert (self.in_height, self.in_width, self.in_channel) == input_frame.shape
        
        start = time.time()
        
#         ps1 = Process(target=self.temporal_job, args=(input_frame, old_frame, 0))        
#         ps1.start()
        ps0 = Process(target=self.spatial_job, args=(input_frame, 0))
        ps0.start()
        
        ps0.join()
#         ps1.join()

#         self.spatial_job(input_frame, 0)
#         self.temporal_job(input_frame, old_frame, 0)
        
        end = time.time()
        
        print(f" elapse {end-start}")
        
        return np.array(self.output[:101])*0.9+np.array(self.output[101:])*0.1 #spatial:(0~101) temporal:(102~202)
    
    