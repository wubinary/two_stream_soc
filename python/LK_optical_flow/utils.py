import cv2, io, PIL
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from IPython.display import  Image, display, clear_output

def your_model(capture_frame):
    # put your model here
    pred_result = np.random.randint(65535)
    return pred_result

def show_frame(capture_frame, pred_result, fps, show_meta):
    
    font = ImageFont.load_default()
    h, w, _ = capture_frame.shape
    if show_meta:
        capture_frame[h-20:, :, :] = 0
    
    # predict value to string
    result = f'result: {pred_result} score: {pred_result} fps:{"{:.1f}".format(fps)}'

    frame = PIL.Image.fromarray(capture_frame)
    if show_meta:
        draw = ImageDraw.Draw(frame)
        draw.text((10,h-15), result, (255,255,255), font=font)
    return frame

def showarray(capture_frame, fps=0, fmt='jpeg', show_meta=True):    
    f = io.BytesIO()
    
    # put your model here
    #pred_result = your_model(capture_frame)
    
    frame = show_frame(capture_frame, 0, fps, show_meta)
    
    clear_output(wait=True)
    frame.save(f, fmt)
    display(Image(data=f.getvalue()))

####################################################################################
    
import numpy as np
from pynq import Xlnk
xlnk = Xlnk()
xlnk.xlnk_reset()

class Feature_bank(object):
    #def __init__(self, ch=20, h=256, w=256, mode='signed'):
    def __init__(self, config):
        (self.ch,self.h,self.w) = (ch,h,w) = (int(config["OpticalFlow"]["channel"]),
                                     int(config["DataConfig"]["image_height"]),int(config["DataConfig"]["image_width"]))
        mode = config["OpticalFlow"]["mode"]
        
        if mode=='signed':
            self.feature_arr = np.zeros(shape=(ch,h,w), dtype=np.int8)
        else:
            self.feature_arr = np.zeros(shape=(ch,h,w), dtype=np.uint8)
        
        self.banks = np.zeros((ch,h,w))
        self.pointer = 0
        
    def push(self, vx, vy):
        banks, pointer = self.banks, self.pointer
        
        banks[pointer] = vx
        banks[pointer+1] = vy
        
        self.pointer = (pointer+2)%self.ch
        
        if self.pointer!=0:
            self.feature_arr[:-self.pointer,:] = banks[self.pointer:]
            self.feature_arr[-self.pointer:,:] = banks[:self.pointer]
        else:
            self.feature_arr[:,:] = banks[:,:]
        
        pass
    
    def get_shared_mem(self) -> xlnk.cma_array:
        return self.feature_arr
    
    def get_np_arr(self) -> np.array:
        return np.moveaxis(self.feature_arr, 0, -1) #(h,w,ch)
    