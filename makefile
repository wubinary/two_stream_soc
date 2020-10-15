SHELL := /bin/zsh 

#######################################################
PART=large #medium

main: hls bitstream

hls:
	#source scripts/hls_lk.sh ${PART}
	source scripts/hls_cnn.sh NLR_Quantize_3 ${PART}

bitstream:
	rm -rf vivado
	vivado -mode batch -source scripts/two_stream_vivado.tcl -nojournal -nolog
	rm -rf .Xil 
	cp vivado/vivado.runs/impl_1/design_1_wrapper.bit design_1.bit
	cp vivado/vivado.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh design_1.hwh

	
