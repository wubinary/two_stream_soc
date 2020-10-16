import numpy as np
import math
import pynq
from pynq import Overlay
from pynq import Xlnk
xlnk = Xlnk()
xlnk.xlnk_reset()

#############################################

IP_BITSTREAM_PATH = './files/design_1.bit'

class LK_accelerator(object):
    hls_lk = None
    dma0 = None
    dma1 = None
        
    #def __init__(self, row=256, col=256, mode='signed', overlay=None):
    def __init__(self, config=None, overlay=None):
        
        self.create_static_pl_instance(overlay) #all LK_accelerator share same PL instance
        
        assert config is not None
               
        self.row, self.col = row, col = int(config["DataConfig"]["image_height"]),int(config["DataConfig"]["image_width"])
        self.mode = mode = config["OpticalFlow"]["mode"]
        
        # create share memory
        self.inp1_arr = xlnk.cma_array(shape=(row,col), dtype=np.uint8)
        self.inp2_arr = xlnk.cma_array(shape=(row,col), dtype=np.uint8)
        self.vx_arr = xlnk.cma_array(shape=(row,col), dtype=np.int8)
        self.vy_arr = xlnk.cma_array(shape=(row,col), dtype=np.int8)
    
    @staticmethod
    def create_static_pl_instance(overlay):
        if overlay is None:
            overlay = Overlay(IP_BITSTREAM_PATH)
        else:
            overlay = overlay
            
        # static PL objects
        LK_accelerator.hls_lk = overlay.hls_LK_0
        LK_accelerator.dma0 = overlay.axi_dma_0 #inp1_arr,vx_arr
        LK_accelerator.dma1 = overlay.axi_dma_1 #inp2_arr,vy_arr
        
    @staticmethod
    def dma_reset():
        # inp1_arr
        LK_accelerator.dma0.sendchannel.stop() 
        LK_accelerator.dma0.sendchannel.start()
        # inp2_arr
        LK_accelerator.dma1.sendchannel.stop() 
        LK_accelerator.dma1.sendchannel.start()
        # vx_arr
        LK_accelerator.dma0.recvchannel.stop() 
        LK_accelerator.dma0.recvchannel.start()
        # vy_arr
        LK_accelerator.dma1.recvchannel.stop()
        LK_accelerator.dma1.recvchannel.start()
        
    def compute(self, img1, img2):
        assert (self.row, self.col)==img1.shape
        assert (self.row, self.col)==img2.shape        
        
        self.inp1_arr[:,:] = img1
        self.inp2_arr[:,:] = img2
        
        # reset dma
        LK_accelerator.dma_reset()
        
        # assign dma memory
        LK_accelerator.dma0.sendchannel.transfer(self.inp1_arr)
        LK_accelerator.dma1.sendchannel.transfer(self.inp2_arr)
        LK_accelerator.dma0.recvchannel.transfer(self.vx_arr)
        LK_accelerator.dma1.recvchannel.transfer(self.vy_arr)
        
        # LK run
        LK_accelerator.hls_lk.write(0x20, self.col) #width
        LK_accelerator.hls_lk.write(0x18, self.row) #height
        LK_accelerator.hls_lk.write(0x00, 0x81)
        
        # dma wait output
        LK_accelerator.dma0.recvchannel.wait()
        LK_accelerator.dma1.recvchannel.wait()

        # output range
        if self.mode=='signed':
            vx = np.array(self.vx_arr)
            vy = np.array(self.vy_arr)
        else:
            vx = (np.array(self.vx_arr).astype(np.int16)+128).astype(np.uint8) #(127,-128)->(255,0)
            vy = (np.array(self.vx_arr).astype(np.int16)+128).astype(np.uint8) #(127,-128)->(255,0)
        return vx, vy
       