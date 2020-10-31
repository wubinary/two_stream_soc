import time, configparser, struct, pickle
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
    def __init__(self, config=None, hardware_instance=None, is_spatial=False):
        
        self.read_config(config, is_spatial)
        
        if hardware_instance is not None:
            self.core0 = hardware_instance

        # allocate buffer for input image
        self.input_buff = xlnk.cma_array(
            shape=(self.buffer_depth, self.WORD_LENGTH),
            dtype=np.uint8)
            
    def read_config(self, config, is_spatial):
        if config is None:
#             print("Initialize configuration...") 
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
        if is_spatial:
            self.img_channel = 16
        else:
            self.img_channel = int(config["DataConfig"]["image_channel"])

        self.WORD_LENGTH = int(ceil(self.data_width/self.precision))
        
        if (self.img_channel > self.Ti) and (self.img_channel % self.Ti != 0):
            self.img_channel += self.Ti - (self.img_channel % self.Ti)

        self.buffer_depth = int(ceil((self.img_channel*self.img_height*self.img_width)/self.WORD_LENGTH))
        print("\t[Info] buff depth",self.buffer_depth)
#         print("Initialize configuration...: done") 
        
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

        if (in_channel > self.Ti) and (in_channel % self.Ti != 0):
            in_channel += self.Ti - (in_channel % self.Ti)

        fm_depth = int(ceil((out_channel*in_height*in_width)/self.WORD_LENGTH))
        wgt_depth = int(ceil((in_channel*out_channel*ker*ker)/self.WORD_LENGTH))

        # ifm_buff = xlnk.cma_array(shape=(ifm_depth, self.WORD_LENGTH), dtype=np.uint8)
        fm_buff = xlnk.cma_array(shape=(fm_depth, self.WORD_LENGTH), dtype=np.uint8)
        wgt_buff = xlnk.cma_array(shape=(wgt_depth, self.WORD_LENGTH), dtype=np.uint8)
        print("memory allocation...: done")

        return fm_buff, wgt_buff

    def setting(self, ofm_buff, ifm_buff, wgt_buff,\
     out_channel, in_channel, in_height, in_width,\
     multiplier, zp_x, zp_w, zp_x_next,\
     ker=3, s=1, poolWin=1):
        
        def float_to_byte(f):
            return struct.pack('f', f)

        self.core0.write(0x10, ifm_buff.physical_address)
        self.core0.write(0x18, ofm_buff.physical_address)
        self.core0.write(0x20, wgt_buff.physical_address)
        self.core0.write(0x28, int(in_height))
        self.core0.write(0x30, int(in_width))
        self.core0.write(0x38, int(in_channel))
        self.core0.write(0x40, int(out_channel))
        # TODO: when input size < 3, this might cause little result
        self.core0.write(0x48, self.Tr if in_height > self.Tr else int(in_height) if in_height > 4 else 4)
        self.core0.write(0x50, self.Tc if in_width > self.Tc else int(in_width) if in_width > 4 else 4)
        self.core0.write(0x58, ker)
        self.core0.write(0x60, s)
        self.core0.write(0x68, poolWin)
        self.core0.write(0x70, float_to_byte(multiplier))
        self.core0.write(0x78, zp_x)
        self.core0.write(0x80, zp_w)
        self.core0.write(0x88, zp_x_next)
        
    def execute(self):
        self.core0.write(0x00, 1)
        isready = self.core0.read(0x00)

        while( isready == 1 ):
            isready = self.core0.read(0x00)

    def init_weight(self, param_path):
        stat_dict = pickle.load(open(param_path, "rb"))
        param_list = list(stat_dict.values())

        l_idx = 0
        for l in self.layers:
#             print("l_idx = ", l_idx, ", l.type = ", l.type)
            if l.type not in ['conv', 'linear']:
                continue
            if l.quantize is True:
                l.weight_data = param_list[l_idx]['qweight'].astype(np.uint8)
                l.multiplier = param_list[l_idx]['scale']
                l.zp_x = param_list[l_idx]['x_zeropoint']
                l.zp_w = param_list[l_idx]['w_zeropoint']
                l.zp_x_next = param_list[l_idx]['xnext_zeropoint']
            else:
                l.weight_data = param_list[l_idx]['qweight']
                l.multiplier = param_list[l_idx]['x_scale']
                l.zp_x = param_list[l_idx]['x_zeropoint']
                
            l_idx+=1
            
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
    
    """
    raw_image: 
        1. img_channel < Ti: channel major, append 0
        2. img_channel > Ti and  img_channel % Ti == 0: 
            channel major
        3. 
    """
    def _convert_raw_image_to_buffer(self, raw_image):
        print("input image shape",raw_image.shape)

        imgH = raw_image.shape[0]
        imgW = raw_image.shape[1]
        img_channel = raw_image.shape[2]
        
        if img_channel < self.WORD_LENGTH:
            zero_padding = np.zeros((imgH, imgW, self.Ti-img_channel), dtype = np.uint8)
            raw_image = np.concatenate((raw_image, zero_padding), axis = 2)

        if img_channel > self.WORD_LENGTH and (img_channel % self.WORD_LENGTH != 0):
            zero_padding = np.zeros((imgH, imgW, self.WORD_LENGTH-(img_channel % self.WORD_LENGTH)), dtype = np.uint8)
            raw_image = np.concatenate((raw_image, zero_padding), axis = 2)

        raw_image = np.transpose(raw_image.reshape((imgH, imgW, int(raw_image.shape[2]/self.Ti), self.Ti)), (2,0,1,3))\
                    .reshape((int(raw_image.shape[2]/self.Ti), imgH, imgW, int((self.Ti/self.WORD_LENGTH)), self.WORD_LENGTH))\
                    .reshape(-1, self.WORD_LENGTH)
        
        print(self.input_buff.shape, raw_image.shape)
        np.copyto(self.input_buff, raw_image)
        
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

        # padding to multiple of 16
        if (in_channel > self.Ti) and (in_channel % self.Ti != 0):
            # (out_channel, in_channel, ker, ker)
            num_res = self.Ti - (in_channel % self.Ti)
            zero_padding = np.zeros((out_channel, num_res,kerH,kerW), dtype=np.uint8)
            wgt = np.concatenate((wgt,zero_padding), axis = 1)

        print("shape of wgt: ", wgt.shape)
        wgt_tmp = np.transpose(\
                    wgt.reshape((int(ceil(out_channel/self.To)), self.To, int(ceil(in_channel/self.Ti)), self.Ti, kerH, kerW)), \
                        (0,2,1,4,5,3))\
                    .reshape((int(ceil(out_channel/self.To)), int(ceil(in_channel/self.Ti)), self.To, kerH, kerW,int(self.Ti/self.WORD_LENGTH),self.WORD_LENGTH))\
                    .reshape(-1,self.WORD_LENGTH)

        np.copyto(layer.wgt_buff, wgt_tmp)
           