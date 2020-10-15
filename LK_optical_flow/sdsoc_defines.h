#ifndef H_LK_SDSOC_H
#define H_LK_SDSOC_H


unsigned long long int sw_sds_counter_total = 0;
unsigned long long int hw_sds_counter_total = 0;
unsigned int sw_sds_counter_num_calls = 0;
unsigned int hw_sds_counter_num_calls = 0;
unsigned long long int sw_sds_counter = 0;
unsigned long long int hw_sds_counter = 0;

#define sw_avg_cpu_cycles() (sw_sds_counter_total / sw_sds_counter_num_calls)
#define hw_avg_cpu_cycles() (hw_sds_counter_total / hw_sds_counter_num_calls)

#ifdef __SDSCC__
#include "sds_lib.h"

#define sw_sds_clk_start() { sw_sds_counter = sds_clock_counter(); sw_sds_counter_num_calls++; }
#define hw_sds_clk_start() { hw_sds_counter = sds_clock_counter(); hw_sds_counter_num_calls++; }
#define sw_sds_clk_stop() { unsigned long long int tmp = sds_clock_counter(); sw_sds_counter_total += tmp-sw_sds_counter; }
#define hw_sds_clk_stop() { unsigned long long int tmp = sds_clock_counter(); hw_sds_counter_total += tmp-hw_sds_counter; }
#define sds_print_results() { int sw_cycles = sw_avg_cpu_cycles();\
	int hw_cycles = hw_avg_cpu_cycles(); printf("Average clock cycles of SW execution: %d\n", sw_cycles);\
	printf("Average clock cycles of HW execution: %d\n", hw_cycles); \
	printf("acceleration factor: = %5.2f\n", ((double) sw_cycles)/((double) hw_cycles) );}

#else // remap as blank lines

#define sw_sds_clk_start()       
#define hw_sds_clk_start()      
#define sw_sds_clk_stop()       
#define hw_sds_clk_stop()       
#define sds_print_results() 

#define sds_alloc malloc
#define sds_free  free

#endif //__SDSCC__


#endif //H_LK_SDSOC_H
