#!/bin/bash


########### Script ###############

HLS_ROOT=$1
SCALE=$2
PROJECT_PATH=$HLS_ROOT/HLS_PROJ

if [ ! -d "$HLS_ROOT" ]; then
	echo "HLS Project ${HLS_ROOT} not exist"
	exit
fi

PART="xczu3eg-sbva484-1-e"
if [ "$2" = "medium" ]; then
        PART="xczu7ev-ffvc1156-2-e"
elif [ "$2" = "large" ]; then
        PART="xczu9eg-ffvb1156-2-e"
fi

cd $HLS_ROOT 
rm -rf $HLS_ROOT/$SCALE
vivado_hls -f script.tcl ./ $SCALE $PART

REPO_PATH=$PROJECT_PATH/repo/$SCALE
IP_PATH=$PROJECT_PATH/$SCALE/impl/ip/xilinx_com_hls_DoCompute_1_0.zip

cd ..

echo $REPO_PATH
if [ ! -d "$REPO_PATH" ]; then
	mkdir -p $REPO_PATH
fi

#cd ..

# unzip ip to repo
unzip -o $IP_PATH -d $REPO_PATH
