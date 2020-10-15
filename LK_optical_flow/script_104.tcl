############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
######################### zcu 104 ##############################
open_project lk
set_top hls_LK
add_files LKof_hls_opt.cpp
#add_files -tb LKof_main.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
#add_files -tb LKof_ref.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
#add_files -tb ap_bmp.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
#add_files -tb motion_compensation.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default
config_export -format ip_catalog -rtl verilog -vivado_phys_opt place -vivado_report_level 0
#source "./lk/solution1/directives.tcl"

#csim_design
csynth_design
#cosim_design
export_design -format ip_catalog
