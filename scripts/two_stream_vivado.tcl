#set PART $argv0 

#  LK HLS implement path <<<<<<<記得改路徑>>>>>>
set hls_lk_impl_path LK_optical_flow/lk/solution1/impl  
# CNN HLS implement path <<<<<<<記得改路徑>>>>>>
set hls_cnn_impl_path NLR_Quantize_3/HLS_PROJ/large/impl
# Vivado project path <<<<<<<記得改路徑>>>>>>
set vivado_path /home/aa/Downloads/two-stream/vivado 

# create new project
create_project vivado ${vivado_path} -part xczu9eg-ffvb1156-2-e -force
# 102
set_property board_part xilinx.com:zcu102:part0:3.3 [current_project]
# 104
#set_property board_part xilinx.com:zcu104:part0:1.1 [current_project]
# z2
#set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]

# create block desin
create_bd_design "design_1"
update_compile_order -fileset sources_1

# [block design] - zynq system
startgroup 
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.3 zynq_ultra_ps_e_0
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ultra_ps_e_0]

# 開 HP 0,1
set_property -dict [list CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__USE__S_AXI_GP1 {1} CONFIG.PSU__USE__S_AXI_GP2 {0}] [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ultra_ps_e_0]

# add LK,CNN IP repository  
#set_property ip_repo_paths ${hls_lk_impl_path} [current_project]
set_property  ip_repo_paths [list ${hls_lk_impl_path} ${hls_cnn_impl_path} ] [current_project]
update_ip_catalog 

# [block design] - DoCompute_0,DoCompute_1
startgroup
create_bd_cell -type ip -vlnv xilinx.com:hls:DoCompute:1.0 DoCompute_0
endgroup 
copy_bd_objs /  [get_bd_cells {DoCompute_0}]

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0 
endgroup 

set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ultra_ps_e_0]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins DoCompute_1/s_axi_CTRL_BUS]
connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_0/M01_AXI] [get_bd_intf_pins DoCompute_0/s_axi_CTRL_BUS]

# dma0,dma1
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
endgroup
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {26} CONFIG.c_sg_include_stscntrl_strm {0} CONFIG.c_m_axis_mm2s_tdata_width {8}] [get_bd_cells axi_dma_0]
copy_bd_objs /  [get_bd_cells {axi_dma_0}]

set_property location {2 530 -857} [get_bd_cells axi_dma_1]
startgroup
set_property -dict [list CONFIG.NUM_MI {4}] [get_bd_cells axi_interconnect_0]
endgroup

connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_0/M02_AXI] [get_bd_intf_pins axi_dma_1/S_AXI_LITE]
connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_0/M03_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

startgroup
set_property -dict [list CONFIG.NUM_MI {5}] [get_bd_cells axi_interconnect_0]
endgroup

startgroup
create_bd_cell -type ip -vlnv xilinx.com:hls:hls_LK:1.0 hls_LK_0
endgroup

connect_bd_intf_net [get_bd_intf_pins hls_LK_0/s_axi_CONTROL_BUS] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/M04_AXI]
connect_bd_intf_net [get_bd_intf_pins hls_LK_0/vx_img] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins hls_LK_0/inp1_img] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
connect_bd_intf_net [get_bd_intf_pins hls_LK_0/vy_img] [get_bd_intf_pins axi_dma_1/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins hls_LK_0/inp2_img] [get_bd_intf_pins axi_dma_1/M_AXIS_MM2S]

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_1
endgroup 

set_property -dict [list CONFIG.NUM_SI {6} CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_1]

connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_1/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_2
endgroup
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_2]

connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_2/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC1_FPD]
connect_bd_intf_net [get_bd_intf_pins DoCompute_0/m_axi_OUTPUT_r] -boundary_type upper [get_bd_intf_pins axi_interconnect_2/S00_AXI]
connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_2/S01_AXI] [get_bd_intf_pins DoCompute_1/m_axi_OUTPUT_r]

startgroup
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins zynq_ultra_ps_e_0/saxihpc1_fpd_aclk]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/DoCompute_0/m_axi_INPUT_r} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} intc_ip {/axi_interconnect_1} master_apm {0}}  [get_bd_intf_pins DoCompute_0/m_axi_INPUT_r]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/DoCompute_1/m_axi_INPUT_r} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} intc_ip {/axi_interconnect_1} master_apm {0}}  [get_bd_intf_pins DoCompute_1/m_axi_INPUT_r]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_0/ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_0/M00_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_0/M01_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_0/M02_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_0/M03_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_0/M04_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/axi_dma_0/M_AXI_MM2S} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} intc_ip {/axi_interconnect_1} master_apm {0}}  [get_bd_intf_pins axi_dma_0/M_AXI_MM2S]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/axi_dma_0/M_AXI_S2MM} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} intc_ip {/axi_interconnect_1} master_apm {0}}  [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/axi_dma_1/M_AXI_MM2S} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} intc_ip {/axi_interconnect_1} master_apm {0}}  [get_bd_intf_pins axi_dma_1/M_AXI_MM2S]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/axi_dma_1/M_AXI_S2MM} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} intc_ip {/axi_interconnect_1} master_apm {0}}  [get_bd_intf_pins axi_dma_1/M_AXI_S2MM]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_2/ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_2/S00_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {Clk "/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)" }  [get_bd_pins axi_interconnect_2/S01_ACLK]
endgroup

assign_bd_address

save_bd_design

# create HDL wrapper
make_wrapper -files [get_files ${vivado_path}/vivado.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ${vivado_path}/vivado.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v

# validate design
validate_bd_design -force

# run syntesis
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# run implement
#launch_runs impl_1 -jobs 3

# generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

#update_compile_order -filset sources_1 

puts "Finish generate bitstream"
