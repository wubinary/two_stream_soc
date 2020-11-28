# Two-stream Action Reconition SOC 💪
###### tags: `powerchip` `ntu` `netdb`
![](https://img.shields.io/static/v1?label=Zynq_UltraScale+&message=zcu102&color=purple)
![](https://img.shields.io/static/v1?label=Board_OS&message=pynq&color=red)
![](https://img.shields.io/static/v1?label=Vivado&message=2019.1&color=orange)
![](https://img.shields.io/static/v1?label=python&message=3.6&color=blue)
![](https://img.shields.io/static/v1?label=shell&message=bash/zsh&color=green)

## Works by
* model : [jeremywu3](https://github.com/jeremywu3) 50%
* cnn acc: [JiaMingLin](https://github.com/JiaMingLin) 50%
* soc: [wubinary](https://github.com/wubinary) 0.01%

cowork power-chips

## Youtube Demo video
[youtube link](https://youtu.be/jTQxzhYSQKI)
[![IMAGE ALT TEXT](https://img.youtube.com/vi/jTQxzhYSQKI/0.jpg)](https://youtu.be/jTQxzhYSQKI "two stream action recognition demo")

## Details
* Platform: Zynq UltraScale+ MPSoC ZCU102 
* Enviroment: Vivado 2019.1 
* Vivado HLS implementations
* System flow:
![](https://i.imgur.com/BMqebcv.gif)

## How to run 💡
```cmd
#### run hls cnn + bitstream
> make

#### run bitstream only
> make bitstream

#### run hls cnn
> make hls
```
