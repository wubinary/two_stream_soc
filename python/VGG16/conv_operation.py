import os
import pynq
import time
import numpy as np
import skimage.measure as skimg

from scipy.signal import convolve2d
from math import floor, ceil
from pynq import Xlnk
from pynq import Overlay

def initIFM(in_channel,inRow,inCol, mode = "ordered"):
    
    if(mode == "ordered"):
        ifm = np.zeros((1, in_channel,inRow,inCol), dtype = np.uint8)
        
        for k in range(in_channel):
            for i in range(inRow):
                for j in range(inCol):
                    ifm[0][k][i][j] = k*inRow*inCol + i*inCol + j
    elif( mode == "random"):
        ifm = np.random.randint(256, size=(1,in_channel,inRow,inCol), dtype=np.uint8)
    elif(mode == "all"):
        ifm = np.ones((1, in_channel,inRow,inCol), dtype = np.uint8)
    elif(mode == "empty"):
        ifm = np.zeros((1, in_channel,inRow,inCol), dtype = np.uint8)

    return ifm

"""
Convert pytorch IFM to Xlnk input
Input: 
    1. ifm: pytorch tensor(batch, in channel, height, width)
Output:
    1. ifm_cma: Xlnk cma(L, WORD_LENGTH), L = (in_channel/Ti)*height*width*(Ti/WORD_LENGTH)
"""
def convertIFM(ifm, Ti, depth, WORD_LENGTH):
    inH = ifm.shape[2]; inW = ifm.shape[3]
    in_channel = ifm.shape[1]
    tmp = np.transpose(ifm[0,:,:,:].reshape((int(in_channel/Ti), Ti, inH, inW)), (0,2,3,1))\
    .reshape((int(in_channel/Ti), inH, inW, int((Ti/WORD_LENGTH)), WORD_LENGTH)).reshape(-1,WORD_LENGTH)

    tmp = np.append(tmp, np.zeros((depth - tmp.shape[0], WORD_LENGTH),np.uint8), axis=0)
    buff = xlnk.cma_array(shape=(depth, WORD_LENGTH), dtype=np.uint8)
    np.copyto(buff, tmp)
    return buff

# currently kernel spatial major, compare with output channel major(word level)
def initWGT(out_channel, in_channel, kerSize, mode = "ordered"):
    if(mode == "ordered"):
        wgt = np.zeros((out_channel, in_channel, kerSize, kerSize), dtype = np.uint8)

        for o in range(out_channel):
            for i in range(in_channel):
                for ky in range(kerSize):
                    for kx in range(kerSize):
                        wgt[o][i][ky][kx] = o*in_channel*kerSize**2 + i*kerSize**2 + ky*kerSize + kx

    elif(mode == "random"):
        #(out_channel, in_channel, ker_height, ker_width)    
        wgt = np.random.randint(256, size=(out_channel, in_channel, kerSize, kerSize), dtype=np.uint8) 
    elif(mode == "all"):
        wgt = np.ones((out_channel, in_channel, kerSize, kerSize), dtype = np.uint8)        

    return wgt

"""
Convert pytorch WGT to Xlnk input
Input:
    1. wgt: pytorch tensor(out channel, in channel, ker_height, ker_width)
Output:
    1. wgt_cma: Xlnk cma(L, WORD_LENGTH), 
        L = (out_channel/To) * (in_channel/Ti) * To * ker_height * ker_width * (Ti/WORD_LENGTH) * WORD_LENGTH
"""
def convertWGT(wgt, To, Ti, kerH, kerW, depth, WORD_LENGTH):
    wgt_tmp = np.transpose(wgt.reshape((int(out_channel/To), To, int(in_channel/Ti), Ti, kerH, kerW)), (0,2,1,4,5,3))\
    .reshape((int(out_channel/To), int(in_channel/Ti), To, kerH, kerW,int(Ti/WORD_LENGTH),WORD_LENGTH))\
    .reshape(-1,WORD_LENGTH)

    wgt_tmp = np.append(wgt_tmp, np.zeros((depth - wgt_tmp.shape[0], WORD_LENGTH),np.uint8), axis=0)
    buff = xlnk.cma_array(shape=(depth, WORD_LENGTH), dtype=np.uint8)
    np.copyto(buff, wgt_tmp)
    return buff
'''
Input:
1. ifm: input feature map, size = (1,in_channel, height, width)
2. wgt: weight, size = (out_channel, in_channel, ker_height, ker_width)

Output:
1. ofm: output feature map, size = (out_row, out_col, out_channel)
'''
def scipy_conv(ifm, wgt, stride):

    # from row major to channel major
    temp_ifm = np.transpose(ifm[0,:,:,:], (1,2,0))
    temp_wgt = np.transpose(wgt, (2,3,0,1))

    in_row = temp_ifm.shape[0]; in_col = temp_ifm.shape[1]; in_channel = temp_ifm.shape[2]
    kerSize = temp_wgt.shape[0]
    padding = floor(kerSize/2)
    out_row = int(ceil(in_row/stride))
    out_col = int(ceil(in_col/stride))
    out_channel = temp_wgt.shape[2]

    tmp = np.zeros((out_row,out_col,in_channel), dtype = np.uint32)
    ofm = np.zeros((out_row, out_col, out_channel), dtype = np.uint8);

    for o in range(out_channel):
        for i in range(in_channel):
            tmp[:,:,i] = \
            convolve2d(np.uint32(temp_ifm[:,:,i]), np.uint32(temp_wgt[:,:,o,i][::-1, ::-1]), \
                mode='same', boundary='fill', fillvalue=0)[::stride, ::stride]
        ofm[:,:,o] = np.sum(tmp, axis = 2)

    # change to row major
    return np.transpose(ofm, (2,0,1))

def sw_pooling(ifm, poolWin):
    channel = ifm.shape[0]; height = ifm.shape[1]; width = ifm.shape[2]
    ofm = np.zeros((channel, int(ceil(height/poolWin)), int(ceil(width/poolWin))), dtype=np.uint8)
    for i in range(channel):
        a = ifm[i,:,:]
        ofm[i,:,:] = skimg.block_reduce(a, (poolWin,poolWin), np.max)
    return ofm

# ifm is row major
def sw_flatten(ifm):
    # flatten ifm
    channel = ifm.shape[0]
    height = ifm.shape[1]
    width = ifm.shape[2]
    flatten_ifm = ifm.reshape((channel * height * width, -1))
    return flatten_ifm

# flatten_ifm: (in_channel, 1)
# wgt shape:(out_channel, in_channel, 1, 1)
def sw_linear(ifm, wgt):
    return wgt.astype(np.float32).dot(ifm.astype(np.float32))

def sw_linear_quant(ifm, wgt, multiplier, zp_x, zp_w, zp_x_next):
    
    output = ((wgt.astype(np.float32)-zp_w).dot((ifm.astype(np.float32)-zp_x)))*multiplier + zp_x_next
    # output = multiplier*np.dot((ifm-zp_x),np.transpose((wgt[:,:,0,0]-zp_w), (1,0)) )+znx
    output = np.round(np.clip(output,0,255)).astype(np.uint8)
    return output


# linear array to row major
def convertOFMOutput(ofm_buff, depth, WORD_LENGTH, out_channel, height, width, Ti):
    # (L, WORD_LENGTH), L = (in_channel/Ti)*height*width*(Ti/WORD_LENGTH)
    ofm = np.zeros((depth, WORD_LENGTH), dtype=np.uint8)
    np.copyto(ofm, ofm_buff)
    ofm = np.transpose(ofm.reshape((int((out_channel/Ti)),height,width,int(Ti/WORD_LENGTH), WORD_LENGTH))\
    .reshape((int((out_channel/Ti)),height,width, Ti)), (0,3,1,2)).reshape((out_channel,height,width))

    return ofm

# outputs are in row major
def compareResult(sw_output, hw_output, channel, height, width):
    err = 0
    str = ""
    for i in range(channel):
        # print("Channel Index = ",i)
        for y in range(height):
            for x in range(width):
                str += "{}: {}, ".format(sw_output[i][y][x], hw_output[i][y][x])
                if sw_output[i][y][x] != hw_output[i][y][x]:
                    err += 1
            # print(str)
            str = ""

    return err


class FPGA_Conv:
    def __init__(self):
        overlay = Overlay("./design_1.bit")
        self.hw_compute = overlay.DoCompute_0


    def hw_conv(self, hw_input, hw_output, hw_wgt, inRow, inCol, \
        in_channel, out_channel, Tr, Tc, kerSize, stride, poolWin):
        self.hw_compute.write(0x10, hw_input.physical_address)
        self.hw_compute.write(0x18, hw_output.physical_address)
        self.hw_compute.write(0x20, hw_wgt.physical_address)
        self.hw_compute.write(0x28, inRow)
        self.hw_compute.write(0x30, inCol)
        self.hw_compute.write(0x38, in_channel)
        self.hw_compute.write(0x40, out_channel)
        self.hw_compute.write(0x48, Tr)
        self.hw_compute.write(0x50, Tc)
        self.hw_compute.write(0x58, kerSize)
        self.hw_compute.write(0x60, stride)
        self.hw_compute.write(0x68, poolWin)

        self.hw_compute.write(0x00, 1)
        isready = self.hw_compute.read(0x00)

        while( isready == 1 ):
            isready = self.hw_compute.read(0x00)


if __name__ == "__main__":
    xlnk = Xlnk()
    xlnk.xlnk_reset()

    in_channel = 16; out_channel = 64
    inRow = 128; inCol = 128
    Tr = 8; Tc = 8
    WORD_LENGTH = 16
    kerSize = 3; stride = 1
    poolWin = 1
    outRow = int(ceil(inRow/poolWin)); outCol= int(ceil(inCol/poolWin))
    ifm_depth = int((in_channel*inRow*inCol)/WORD_LENGTH)#300000
    ofm_depth = int((out_channel*outRow*outCol)/WORD_LENGTH)#300000
    wgt_depth = int((in_channel*out_channel*kerSize*kerSize)/WORD_LENGTH)#150000 # for only one layer
    To = 16; Ti = 16

    print("Allocating memory...")
    ifm = initIFM(in_channel,inRow,inCol, mode = "random")  # row major
    ofm = initIFM(out_channel,outRow,outCol, mode = "empty")
    wgt = initWGT(out_channel, in_channel, kerSize, mode = "random")

    ifm_buff = convertIFM(ifm, Ti, ifm_depth, WORD_LENGTH)
    ofm_buff = convertIFM(ofm, To, ofm_depth, WORD_LENGTH)
    wgt_buff = convertWGT(wgt, To, Ti, kerSize, kerSize, wgt_depth, WORD_LENGTH)

    print("Allocating memory..., Done")

    print("Initialize Harware")
    HWConv = FPGA_Conv()
    
    print("Docompute Convolution")

    # hardware
    print("Hardware Compute...")
    hw_begin = time.perf_counter()
    HWConv.hw_conv(ifm_buff, ofm_buff, wgt_buff, inRow, inCol, \
        in_channel, out_channel, Tr, Tc, kerSize, stride, poolWin)
    print("Hardware Time(ms): ", (time.perf_counter() - hw_begin)*1000)

    # software
    print("Software Compute...")
    sw_begin = time.perf_counter() 
    scipy_result = scipy_conv(ifm, wgt, stride)
    # scipy_result = sw_pooling(sw_ofm,poolWin)
    # flatten_ifm = sw_flatten(ifm)
    # scipy_result = sw_linear(flatten_ifm, wgt)
    # sw_result = np.expand_dims(scipy_result, axis = 2)
    print("Software Time(ms): ", (time.perf_counter() - sw_begin)*1000)

    print("Conpare Result")
    hw_ofm = convertOFMOutput(ofm_buff, ofm_depth, WORD_LENGTH, out_channel, outRow, outCol, Ti)
    # print(ofm_buff)
    print(scipy_result.shape, hw_ofm.shape)
    err = compareResult(scipy_result, hw_ofm, out_channel, outRow, outCol)
    print(err)
