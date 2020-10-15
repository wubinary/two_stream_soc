#!/bin/zsh

Red='\033[0;31m'
Cyan='\033[1;36m'
NC='\033[0m' # No Color

####### start script #######

SCALE=$1

case $SCALE in 
	small) SCRIPT="script_z2.tcl" ;;
	medium) SCRIPT="script_104.tcl" ;;
	large) SCRIPT="script_102.tcl" ;;
	*) SCRIPT="xxx.tcl" ;;
esac 

cd LK_optical_flow/ 

echo "${Cyan}vivado_hls $SCRIPT $NC"
vivado_hls $SCRIPT 

cd ..

