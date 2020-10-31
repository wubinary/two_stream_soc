import cv2, io, PIL
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from IPython.display import  Image, display, clear_output

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
    
    
######################################################################  

def your_model(capture_frame):
    # put your model here
    pred_result = np.random.randint(65535)
    return pred_result

def show_frame(capture_frame, pred_result, fps, show_meta, topk=None):
    
    font = ImageFont.load_default()
    h, w, _ = capture_frame.shape
    if show_meta:
        capture_frame[h-20:, :, :] = 0
    
    # predict value to string
    topk_class = []
    class_str = ""
    for c in topk:
        topk_class.append(f"{action_map[c]}")
        #class_str += f"{action_map[c]} "
    topk_class.sort()
    for c in topk_class:
        class_str += f"{c} "
    result = f'result: {pred_result} score: {pred_result} fps:{"{:.1f}".format(fps)} actions:{class_str}'

    frame = PIL.Image.fromarray(capture_frame)
    if show_meta:
        draw = ImageDraw.Draw(frame)
        draw.text((10,h-15), result, (255,255,255), font=font)
    return frame

def showarray(capture_frame, fps=0, fmt='jpeg', show_meta=True, topk=None):    
    f = io.BytesIO()
    
    # put your model here
    #pred_result = your_model(capture_frame)
    
    frame = show_frame(capture_frame, 0, fps, show_meta, topk)
    
    clear_output(wait=True)
    frame.save(f, fmt)
    display(Image(data=f.getvalue()))
    
# dictionary of actions
action_map = {
1 : 'ApplyEyeMakeup',
2 :'ApplyLipstick',
3 :'Archery',
4 :'BabyCrawling',
5 :'BalanceBeam',
6 :'BandMarching',
7 :'BaseballPitch',
8 :'Basketball',
9 :'BasketballDunk',
10 :'BenchPress',
11 :'Biking',
12 :'Billiards',
13 :'BlowDryHair',
14 :'BlowingCandles',
15 :'BodyWeightSquats',
16 :'Bowling',
17 :'BoxingPunchingBag',
18 :'BoxingSpeedBag',
19 :'BreastStroke',
20 :'BrushingTeeth',
21 :'CleanAndJerk',
22 :'CliffDiving',
23 :'CricketBowling',
24 :'CricketShot',
25 :'CuttingInKitchen',
26 :'Diving',
27 :'Drumming',
28 :'Fencing',
29 :'FieldHockeyPenalty',
30 :'FloorGymnastics',
31 :'FrisbeeCatch',
32 :'FrontCrawl',
33 :'GolfSwing',
34 :'Haircut',
35 :'Hammering',
36 :'HammerThrow',
37 :'HandstandPushups',
38 :'HandstandWalking',
39 :'HeadMassage',
40 :'HighJump',
41 :'HorseRace',
42 :'HorseRiding',
43 :'HulaHoop',
44 :'IceDancing',
45 :'JavelinThrow',
46 :'JugglingBalls',
47 :'JumpingJack',
48 :'JumpRope',
49 :'Kayaking',
50 :'Knitting',
51 :'LongJump',
52 :'Lunges',
53 :'MilitaryParade',
54 :'Mixing',
55 :'MoppingFloor',
56 :'Nunchucks',
57 :'ParallelBars',
58 :'PizzaTossing',
59 :'PlayingCello',
60 :'PlayingDaf',
61 :'PlayingDhol',
62 :'PlayingFlute',
63 :'PlayingGuitar',
64 :'PlayingPiano',
65 :'PlayingSitar',
66 :'PlayingTabla',
67 :'PlayingViolin',
68 :'PoleVault',
69 :'PommelHorse',
70 :'PullUps',
71 :'Punch',
72 :'PushUps',
73 :'Rafting',
74 :'RockClimbingIndoor',
75 :'RopeClimbing',
76 :'Rowing',
77 :'SalsaSpin',
78 :'ShavingBeard',
79 :'Shotput',
80 :'SkateBoarding',
81 :'Skiing',
82 :'Skijet',
83 :'SkyDiving',
84 :'SoccerJuggling',
85 :'SoccerPenalty',
86 :'StillRings',
87 :'SumoWrestling',
88 :'Surfing',
89 :'Swing',
90 :'TableTennisShot',
91 :'TaiChi',
92 :'TennisSwing',
93 :'ThrowDiscus',
94 :'TrampolineJumping',
95 :'Typing',
96 :'UnevenBars',
97 :'VolleyballSpiking',
98 :'WalkingWithDog',
99 :'WallPushups',
100 :'WritingOnBoard',
101 :'YoYo'}
