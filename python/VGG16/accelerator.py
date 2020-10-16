import time, configparser
import numpy as np

from math import floor, ceil

from pynq import Xlnk
xlnk = Xlnk()
xlnk.xlnk_reset()

#############################################

IP_BITSTREAM_PATH = './files/design_1.bit'

class LK_accelerator(object):
    hls_lk = None
    dma0 = None
    dma1 = None
        
    def __init__(self, row=256, col=256, mode='signed', overlay=None):
        self.create_static_pl_instance(overlay) #all LK_accelerator share same PL instance
        
        self.row, self.col = row, col
        self.mode = mode
        
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

###############################################################################################################

class CNN_accelerator(object):
        
    def __init__(self, config=None, hardware_instance=None):
        
        self.read_config(config)
        
        if hardware_instance is not None:
            self.core0 = hardware_instance

        # allocate buffer for input image
        self.input_buff = xlnk.cma_array(
            shape=(self.buffer_depth, self.WORD_LENGTH),
            dtype=np.uint8)
            
    def read_config(self, config):
        if config is None:
            config = configparser.ConfigParser()
            config.read('./files/config.config')
            
        self.board = config["FPGAConfig"]["name"]
        self.bitstream_path = config["FPGAConfig"]["bitstream_path"]
        self.precision = int(config["PEConfig"]["precision"])
        self.data_width = int(config["PEConfig"]["data_width"])
        self.Ti = int(config["PEConfig"]["Ti"])
        self.To = int(config["PEConfig"]["To"])
        self.Tr = int(config["PEConfig"]["Tr"])
        self.Tc = int(config["PEConfig"]["Tc"])

        self.img_height = int(config["DataConfig"]["image_height"])
        self.img_width = int(config["DataConfig"]["image_width"])
        self.img_channel = int(config["DataConfig"]["image_channel"])

        self.WORD_LENGTH = int(ceil(self.data_width/self.precision))
        self.buffer_depth = int(ceil((self.img_channel*self.img_height*self.img_width)/self.WORD_LENGTH))
        print("Initialize configuration...: done") 
        
    def __call__(self, raw_image):
        self._convert_raw_image_to_buffer(raw_image)
        print("executing layers...")

        hw_begin = time.perf_counter()
        fm = self.layers[0](self.input_buff)
        print("layer {}, IFM size = {}, WGT size = {}, OFM size = {}, time = {}(ms)"\
            .format(0, \
                (self.layers[0].in_height, self.layers[0].in_width, self.layers[0].in_channel),\
                self.layers[0].weight_shape, \
                (self.layers[0].out_height, self.layers[0].out_width, self.layers[0].out_channel),\
                (time.perf_counter() - hw_begin)*1000))

        cnt = 1
        for l in self.layers[1:]:
            hw_begin = time.perf_counter()
            fm = l(fm)
            print("layer {}, IFM size = {}, WGT size = {}, OFM size = {}, time = {}(ms)"\
            .format(cnt, \
                (l.in_height, l.in_width, l.in_channel),\
                l.weight_shape, \
                (l.out_height, l.out_width, l.out_channel),\
                (time.perf_counter() - hw_begin)*1000))
            cnt+=1

        return fm

    def mem_alloc(self, out_channel, in_channel, in_height, in_width, ker):
        print("memory allocation...")
        # ifm_depth = int(ceil((in_channel*in_height*in_width)/self.WORD_LENGTH))
        if (out_channel % self.To != 0):
            out_channel += self.To - (out_channel % self.To)

        fm_depth = int(ceil((out_channel*in_height*in_width)/self.WORD_LENGTH))
        wgt_depth = int(ceil((in_channel*out_channel*ker*ker)/self.WORD_LENGTH))

        # ifm_buff = xlnk.cma_array(shape=(ifm_depth, self.WORD_LENGTH), dtype=np.uint8)
        fm_buff = xlnk.cma_array(shape=(fm_depth, self.WORD_LENGTH), dtype=np.uint8)
        wgt_buff = xlnk.cma_array(shape=(wgt_depth, self.WORD_LENGTH), dtype=np.uint8)
        #print("memory allocation...: done")

        return fm_buff, wgt_buff

    def setting(self, ofm_buff, ifm_buff, wgt_buff,
                out_channel, in_channel, in_height, in_width, ker=3, s=1, poolWin=1):

        self.core0.write(0x10, ifm_buff.physical_address)
        self.core0.write(0x18, ofm_buff.physical_address)
        self.core0.write(0x20, wgt_buff.physical_address)
        self.core0.write(0x28, int(in_height))
        self.core0.write(0x30, int(in_width))
        self.core0.write(0x38, int(in_channel))
        self.core0.write(0x40, int(out_channel))
        self.core0.write(0x48, self.Tr)
        self.core0.write(0x50, self.Tc)
        self.core0.write(0x58, ker)
        self.core0.write(0x60, s)
        self.core0.write(0x68, poolWin)
        
    def execute(self):
        self.core0.write(0x00, 1)
        isready = self.core0.read(0x00)

        while( isready == 1 ):
            isready = self.core0.read(0x00)
            
    def load_parameters(self):
        if self.layers is None:
            raise("Network layers are not initialized")

        for l in self.layers:
            if l.type is not "conv":
                continue

            if l.weight_data.shape != l.weight_shape:
                raise("Input weight shape is " + l.weight_data.shape \
                    + ", not match with setting " + l.weight_shape)

            self._convert_weight_to_buffer(l)

        return 0
    
    def _convert_raw_image_to_buffer(self, raw_image):
        imgH = raw_image.shape[0]
        imgW = raw_image.shape[1]
        img_channel = raw_image.shape[2]
        
        if img_channel < self.Ti:
            zero_padding = np.zeros((imgH, imgW, self.Ti-img_channel), dtype = np.uint8)
            raw_image = np.concatenate((raw_image, zero_padding), axis = 2)

        np.copyto(self.input_buff, raw_image.reshape(-1,raw_image.shape[2]))
        
    """
    Convert pytorch WGT to Xlnk input
    Input:
        1. wgt: pytorch tensor(out channel, in channel, ker_height, ker_width)
    Output:
        1. wgt_cma: Xlnk cma(Depth, WORD_LENGTH), 
            Depth = (out_channel/To) * (in_channel/Ti) * To * ker_height * ker_width * (Ti/WORD_LENGTH) * WORD_LENGTH
    """
    def _convert_weight_to_buffer(self, layer):
        wgt = layer.weight_data
        out_channel = layer.out_channel
        in_channel = layer.in_channel
        kerH = layer.ker
        kerW = layer.ker

        if in_channel < self.Ti:
            zero_padding = np.zeros((out_channel,self.Ti - in_channel,kerH,kerW), dtype=np.uint8)
            wgt = np.concatenate((wgt,zero_padding), axis = 1)

        # if (out_channel % self.To != 0):
        #   zero_padding = np.zeros((self.To - (out_channel % self.To),in_channel,kerH,kerW), dtype=np.uint8)
        #   wgt = np.concatenate((wgt,zero_padding), axis = 0)

        print("shape of wgt: ", wgt.shape)
        wgt_tmp = np.transpose(\
            wgt.reshape((int(ceil(out_channel/self.To)), self.To, int(ceil(in_channel/self.Ti)), self.Ti, kerH, kerW)), \
                (0,2,1,4,5,3))\
                .reshape((int(ceil(out_channel/self.To)), int(ceil(in_channel/self.Ti)), self.To, kerH, kerW,int(self.Ti/self.WORD_LENGTH),self.WORD_LENGTH))\
                .reshape(-1,self.WORD_LENGTH)

        np.copyto(layer.wgt_buff, wgt_tmp)
           