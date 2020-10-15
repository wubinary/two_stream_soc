/*************************************************************************************
Vendor:					Xilinx 
Associated Filename:	LKof_hls_opt.cpp
Purpose:				Non Iterative Lukas Kanade Optical Flow
Revision History:		31 August 2016 - final release
author:					daniele.bagni@xilinx.com        

based on http://uk.mathworks.com/help/vision/ref/opticalflowlk-class.html?searchHighlight=opticalFlowLK%20class

**************************************************************************************
© Copyright 2008 - 2016 Xilinx, Inc. All rights reserved. 

This file contains confidential and proprietary information of Xilinx, Inc. and 
is protected under U.S. and international copyright and other intellectual 
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials 
distributed herewith. Except as otherwise provided in a valid license issued to 
you by Xilinx, and to the maximum extent permitted by applicable law: 
(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX 
HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, 
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR 
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether 
in contract or tort, including negligence, or under any other theory of 
liability) for any loss or damage of any kind or nature related to, arising under 
or in connection with these materials, including for any direct, or any indirect, 
special, incidental, or consequential loss or damage (including loss of data, 
profits, goodwill, or any type of loss or damage suffered as a result of any 
action brought by a third party) even if such damage or loss was reasonably 
foreseeable or Xilinx had been advised of the possibility of the same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any 
application requiring fail-safe performance, such as life-support or safety 
devices or systems, Class III medical devices, nuclear facilities, applications 
related to the deployment of airbags, or any other applications that could lead 
to death, personal injury, or severe property or environmental damage 
(individually and collectively, "Critical Applications"). Customer assumes the 
sole risk and liability of any use of Xilinx products in Critical Applications, 
subject only to applicable laws and regulations governing limitations on product 
liability. 

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT 
ALL TIMES.

*******************************************************************************/

#include "LKof_defines.h"

bool hls_matrix_inversion(sum_t A[2][2], sum_t B[2], int threshold, float &Vx, float &Vy)
{
	bool invertible = 0;
	sum_t inv_A[2][2];
	sum_t a, b, c, d; 
	det_t det_A, abs_det_A, neg_det_A, zero = 0;
	float recipr_det_A; 

	a = A[0][0]; b = A[0][1]; c = A[1][0]; d = A[1][1];
	
	sum2_t a_x_d, b_x_c, mult1, mult2, mult3, mult4;
	det_t t_Vx, t_Vy;

	//det_A = (a*d)-(b*c); //determinant of matrix  A = [a b; c d]
	a_x_d = (sum2_t) a * (sum2_t) d;
	b_x_c = (sum2_t) b * (sum2_t) c;
	det_A = a_x_d - b_x_c;
	neg_det_A = (zero-det_A);
	abs_det_A = (det_A > zero) ? det_A : neg_det_A;
	
	recipr_det_A = (1.0f)/det_A;

	//compute the inverse of matrix A anyway even if it is not invertible: inv_A = [d -b; -c a]/det_A
	// note that 1/det_A is done only at the last instruction to save resources
	if (det_A == 0) recipr_det_A = 0;
	inv_A[0][0] =  d;
	inv_A[0][1] = -b;
	inv_A[1][0] = -c;
	inv_A[1][1] =  a;

	//solve the matrix equation: [Vx Vy] = -inv_A[] * B[]
	//Vx = -(inv_A[0][0] * B[0] + inv_A[0][1] * B[1]);
	//Vy = -(inv_A[1][0] * B[0] + inv_A[1][1] * B[1]);
	mult1 = (sum2_t) inv_A[0][0] * (sum2_t) B[0];
	mult2 = (sum2_t) inv_A[0][1] * (sum2_t) B[1];
	mult3 = mult2; //(sum2_t) inv_A[1][0] * (sum2_t) B[0];
	mult4 = (sum2_t) inv_A[1][1] * (sum2_t) B[1];
	t_Vx = -(mult1 + mult2);
	t_Vy = -(mult3 + mult4);

	Vx = t_Vx * recipr_det_A;
	Vy = t_Vy * recipr_det_A;

	if (det_A == 0) // zero input pixels
	{
		invertible = 0;
		Vx = 0; Vy = 0;
	}
	else if (abs_det_A < threshold) // the matrix is not invertible
	{
		invertible = 0;
		Vx = 0; Vy = 0;
	}
	else
	{
		invertible = 1;
	}

	return invertible;

}



#ifdef ISOTROPIC_NOT_OPTIMIZED

pix_t hls_isotropic_kernel(pix_t window[FILTER_SIZE*FILTER_SIZE])
{
	
	// isotropic smoothing filter 5x5
	const coe_t coeff[FILTER_SIZE][FILTER_SIZE] = {
		{ 1,    4,    6,    4,    1},
		{ 4,   16,   24,   16,    4},
		{ 6,   24,   36,   24,    6},
		{ 4,   16,   24,   16,    4},
		{ 1,    4,    6,    4,    1}
	};

	// local variables
	int accum = 0;
	int normalized_accum;
	pix_t final_val;
	unsigned char i, j;

	//Compute the 2D convolution
	L1:for (i = 0; i < FILTER_SIZE; i++) {
		L2:for (j = 0; j < FILTER_SIZE; j++) {

			ap_int<(BITS_PER_COEFF+BITS_PER_PIXEL)> loc_mult = ( window[i*FILTER_SIZE+j] * coeff[i][j]);

			accum = accum + loc_mult;
		}
	}

	// do the correct normalization if needed
	normalized_accum = accum / 256;

	final_val = (pix_t) normalized_accum;

	return final_val;
}

#ifdef __SYNTHESIS__
void hls_twoIsotropicFilters(AXI_STREAM_U& inp1_img, AXI_STREAM_U& inp2_img,
						 	 pix_t out1_img[MAX_HEIGHT*MAX_WIDTH], pix_t out2_img[MAX_HEIGHT*MAX_WIDTH],
								 unsigned short int height, unsigned short int width)
{
#else
void hls_twoIsotropicFilters(unsigned short int inp1_img[MAX_HEIGHT*MAX_WIDTH], unsigned short int  inp2_img[MAX_HEIGHT*MAX_WIDTH],
						 	 pix_t out1_img[MAX_HEIGHT*MAX_WIDTH], pix_t out2_img[MAX_HEIGHT*MAX_WIDTH],
								 unsigned short int height, unsigned short int width)
{
#endif

	//static int hls_cnt_read=0;
	//static int hls_cnt_write=0;

	short int row, col;
	pix_t filt_out1, filt_out2, pix1, pix2;

	pix_t pixel1[FILTER_SIZE], pixel2[FILTER_SIZE];

	pix_t window1[FILTER_SIZE*FILTER_SIZE];
	#pragma HLS ARRAY_PARTITION variable=window1 complete dim=0
	pix_t window2[FILTER_SIZE*FILTER_SIZE];
	#pragma HLS ARRAY_PARTITION variable=window2 complete dim=0

	static pix_t lpf1_line_buffer[FILTER_SIZE][MAX_WIDTH];
	#pragma HLS ARRAY_PARTITION variable=lpf1_line_buffer complete dim=1
	static pix_t lpf2_line_buffer[FILTER_SIZE][MAX_WIDTH];
	#pragma HLS ARRAY_PARTITION variable=lpf2_line_buffer complete dim=1

   // effective filtering
	L1: for(row = 0; row < height+FILTER_OFFS; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT  max=480
		L2: for(col = 0; col < width+FILTER_OFFS; col++)
		{
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_TRIPCOUNT max=640


			// Line Buffer fill
			if(col < width)
				for(unsigned char ii = 0; ii < FILTER_SIZE-1; ii++)
				{
					pixel1[ii] = lpf1_line_buffer[ii][col] = lpf1_line_buffer[ii+1][col];
					pixel2[ii] = lpf2_line_buffer[ii][col] = lpf2_line_buffer[ii+1][col];
				}

			//There is an offset to accomodate the active pixel region
			if((col < width) && (row < height))
			{
				//hls_cnt_read++;
				#ifdef __SYNTHESIS__
				pix1 = (unsigned short int ) inp1_img.read().data;
				pix2 = (unsigned short int ) inp2_img.read().data;
				#else 
				pix1 = inp1_img[row*MAX_WIDTH+col];
			    pix2 = inp2_img[row*MAX_WIDTH+col];
				#endif
				pixel1[FILTER_SIZE-1] = lpf1_line_buffer[FILTER_SIZE-1][col] = pix1;
				pixel2[FILTER_SIZE-1] = lpf2_line_buffer[FILTER_SIZE-1][col] = pix2;
			}

			//Shift right the processing window to make room for the new column
			L3:for(unsigned char ii = 0; ii < FILTER_SIZE; ii++)
				L4:for(unsigned char jj = 0; jj < FILTER_SIZE-1; jj++)
				{
					window1[ii*FILTER_SIZE+jj] = window1[ii*FILTER_SIZE+jj+1];
					window2[ii*FILTER_SIZE+jj] = window2[ii*FILTER_SIZE+jj+1];
				}
			L5:for(unsigned char ii = 0; ii < FILTER_SIZE; ii++)
			{
				window1[ii*FILTER_SIZE+FILTER_SIZE-1] = pixel1[ii];
				window2[ii*FILTER_SIZE+FILTER_SIZE-1] = pixel2[ii];
			}
			//This design assumes there are no edges on the boundary of the image
			if ( (row >= FILTER_OFFS) & (col >= FILTER_OFFS) &  (row < height)  & (col< width) )
			{
				   filt_out1 = hls_isotropic_kernel(window1);
				   filt_out2 = hls_isotropic_kernel(window2);
			}
			else
			{
				   filt_out1 = 0;
				   filt_out2 = 0;
			}


			if ( (row >= FILTER_OFFS) & (col >= FILTER_OFFS)) // &  (row < height)  & (col< width) )
			{
				//hls_cnt_write++;
				   out1_img[(row-FILTER_OFFS)*MAX_WIDTH+(col-FILTER_OFFS)] = (filt_out1);
				   out2_img[(row-FILTER_OFFS)*MAX_WIDTH+(col-FILTER_OFFS)] = (filt_out2);
			}


		} // end of L2
	} // end of L1



}

#else

dualpix_t hls_isotropic_kernel(dualpix_t window[FILTER_SIZE*FILTER_SIZE])
{

	// isotropic smoothing filter 5x5
	const coe_t coeff[FILTER_SIZE][FILTER_SIZE] = {
		{ 1,    4,    6,    4,    1},
		{ 4,   16,   24,   16,    4},
		{ 6,   24,   36,   24,    6},
		{ 4,   16,   24,   16,    4},
		{ 1,    4,    6,    4,    1}
	};

	// local variables
	int accum1 = 0;
	int normalized_accum1;
	int accum2 = 0;
	int normalized_accum2;
	pix_t pix1, pix2, final_val1, final_val2;
	unsigned char i, j;
	dualpix_t two_pixels;

	//Compute the 2D convolution
	L1:for (i = 0; i < FILTER_SIZE; i++) {
		L2:for (j = 0; j < FILTER_SIZE; j++) {
			two_pixels = window[i*FILTER_SIZE+j];
			pix1 = two_pixels(  BITS_PER_PIXEL-1,              0);
			pix2 = two_pixels(2*BITS_PER_PIXEL-1, BITS_PER_PIXEL);

			ap_int<(BITS_PER_COEFF+BITS_PER_PIXEL)> loc_mult1 = ( pix1 * coeff[i][j]);
			ap_int<(BITS_PER_COEFF+BITS_PER_PIXEL)> loc_mult2 = ( pix2 * coeff[i][j]);

			accum1 = accum1 + loc_mult1;
			accum2 = accum2 + loc_mult2;

		}
	}

	// do the correct normalization if needed
	normalized_accum1 = accum1 / 256;
	normalized_accum2 = accum2 / 256;
	final_val1 = (pix_t) normalized_accum1;
	final_val2 = (pix_t) normalized_accum2;

	two_pixels(  BITS_PER_PIXEL-1,              0) = final_val1;
	two_pixels(2*BITS_PER_PIXEL-1, BITS_PER_PIXEL) = final_val2;

	return two_pixels;
}

#ifdef __SYNTHESIS__
void hls_twoIsotropicFilters(AXI_STREAM_U& inp1_img, AXI_STREAM_U& inp2_img,
						 	 pix_t out1_img[MAX_HEIGHT*MAX_WIDTH], pix_t out2_img[MAX_HEIGHT*MAX_WIDTH],
								 unsigned short int height, unsigned short int width)
{
#else
void hls_twoIsotropicFilters(unsigned short int inp1_img[MAX_HEIGHT*MAX_WIDTH], unsigned short int  inp2_img[MAX_HEIGHT*MAX_WIDTH],
						 	 pix_t out1_img[MAX_HEIGHT*MAX_WIDTH], pix_t out2_img[MAX_HEIGHT*MAX_WIDTH],
								 unsigned short int height, unsigned short int width)
{
#endif

	unsigned short int row, col;

	pix_t filt_out1, filt_out2, pix1, pix2;
	dualpix_t two_pixels;

	dualpix_t pixels[FILTER_SIZE];
    #pragma HLS ARRAY_PARTITION variable=pixels complete dim=0

	dualpix_t window[FILTER_SIZE*FILTER_SIZE];
	#pragma HLS ARRAY_PARTITION variable=window complete dim=0

	static dualpix_t lpf_lines_buffer[FILTER_SIZE][MAX_WIDTH];
	#pragma HLS ARRAY_PARTITION variable=lpf_lines_buffer complete dim=1

  // effective filtering
	L1: for(row = 0; row < height+FILTER_OFFS; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT  min=hls_MIN_H max=hls_MAX_H
		L2: for(col = 0; col < width+FILTER_OFFS; col++)
		{
			#pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT min=hls_MIN_W max=hls_MAX_W

			// Line Buffer fill
			if(col < width)
				for(unsigned char ii = 0; ii < FILTER_SIZE-1; ii++)
				{
					pixels[ii] = lpf_lines_buffer[ii][col] = lpf_lines_buffer[ii+1][col];
				}

			//There is an offset to accomodate the active pixel region
			if((col < width) && (row < height))
			{

				//hls_cnt_read++;
				#ifdef __SYNTHESIS__
				pix1 = (pix_t) inp1_img.read().data;
				pix2 = (pix_t) inp2_img.read().data;
				#else 
				pix1 = (pix_t) inp1_img[row*MAX_WIDTH+col];
			    pix2 = (pix_t) inp2_img[row*MAX_WIDTH+col];
				#endif
				two_pixels(2*BITS_PER_PIXEL-1, BITS_PER_PIXEL) = pix2;
				two_pixels(  BITS_PER_PIXEL-1,              0) = pix1;

				pixels[FILTER_SIZE-1] = lpf_lines_buffer[FILTER_SIZE-1][col] = two_pixels;
			}

			//Shift right the processing window to make room for the new column
			L3:for(unsigned char ii = 0; ii < FILTER_SIZE; ii++)
				L4:for(unsigned char jj = 0; jj < FILTER_SIZE-1; jj++)
				{
					window[ii*FILTER_SIZE+jj] = window[ii*FILTER_SIZE+jj+1];
				}
			L5:for(unsigned char ii = 0; ii < FILTER_SIZE; ii++)
			{
				window[ii*FILTER_SIZE+FILTER_SIZE-1] = pixels[ii];
			}
			//This design assumes there are no edges on the boundary of the image
			if ( (row >= FILTER_OFFS) & (col >= FILTER_OFFS) &  (row < height)  & (col< width) )
			{
				   two_pixels = hls_isotropic_kernel(window);
				   filt_out1 = two_pixels(  BITS_PER_PIXEL-1,              0);
				   filt_out2 = two_pixels(2*BITS_PER_PIXEL-1, BITS_PER_PIXEL);
			}
			else
			{
				   filt_out1 = 0;
				   filt_out2 = 0;
			}

			if ( (row >= FILTER_OFFS) & (col >= FILTER_OFFS)) 
			{
				   out1_img[(row-FILTER_OFFS)*MAX_WIDTH+(col-FILTER_OFFS)] = (filt_out1);
				   out2_img[(row-FILTER_OFFS)*MAX_WIDTH+(col-FILTER_OFFS)] = (filt_out2);
			}


		} // end of L2
	} // end of L1


}
#endif



void hls_derivatives_kernel(dualpix_t window[FILTER_SIZE*FILTER_SIZE], flt_t &Ix, flt_t &Iy, flt_t &It)
{
	
	// derivative filter in a 5x5 kernel size  [-1 8 0 -8 1]^T
	// coefficients are swapped to get same results as MATLAB
	const coe_t y_coeff[FILTER_SIZE][FILTER_SIZE] = {
		{ 0,    0,    1,    0,    0},
		{ 0,    0,   -8,    0,    0},
		{ 0,    0,    0,    0,    0},
		{ 0,    0,    8,    0,    0},
		{ 0,    0,   -1,    0,    0}
	};

	// derivative filter in a 5x5 kernel size: [-1 8 0 -8 1]
	// coefficients are swapped to get same results as MATLAB
	const coe_t x_coeff[FILTER_SIZE][FILTER_SIZE] = {
		{ 0,    0,    0,    0,    0},
		{ 0,    0,    0,    0,    0},
		{ 1,   -8,    0,    8,   -1},
		{ 0,    0,    0,    0,    0},
		{ 0,    0,    0,    0,    0}
	};

	// local variables
	int accum_x = 0;
	int accum_y = 0;

	dualpix_t two_pix;
	pix_t pix1, pix2;
	int normalized_accum_x, normalized_accum_y;
	flt_t final_val_x, final_val_y, final_val_t;
	unsigned char i, j;

	//Compute the 2D convolution
	L1:for (i = 0; i < FILTER_SIZE; i++) 
    {
		L2:for (j = 0; j < FILTER_SIZE; j++) 
		{
			two_pix = window[i*FILTER_SIZE+j]; 
			pix1 = two_pix(2*BITS_PER_PIXEL-1, BITS_PER_PIXEL);
			pix2 = two_pix(  BITS_PER_PIXEL-1, 0);

			signed short int loc_mult_x = ( pix1 * x_coeff[i][j]); 
			accum_x = accum_x + loc_mult_x;
			signed short int loc_mult_y = ( pix1 * y_coeff[i][j]); 
			accum_y = accum_y + loc_mult_y;

			if ( (i==2)&(j==2) )
				final_val_t = pix2 - pix1; //central pix is window[2][2]
		}
	}

	// do the correct normalization if needed
	normalized_accum_x = accum_x / 12;
	normalized_accum_y = accum_y / 12;

	final_val_x = (flt_t) normalized_accum_x;
	final_val_y = (flt_t) normalized_accum_y;

	Ix = final_val_x;
	Iy = final_val_y;
	It = final_val_t;

}


void hls_SpatialTemporalDerivatives(pix_t   inp1_img[MAX_HEIGHT*MAX_WIDTH],
									pix_t   inp2_img[MAX_HEIGHT*MAX_WIDTH],
									flt_t out_Ix_img[MAX_HEIGHT*MAX_WIDTH], 
									flt_t out_Iy_img[MAX_HEIGHT*MAX_WIDTH], 
									flt_t out_It_img[MAX_HEIGHT*MAX_WIDTH], 
									unsigned short int height, unsigned short int width)
{

	unsigned short int row, col;
	flt_t filt_out_x, filt_out_y, filt_out_t;
	
	dualpix_t two_pix, two_pixel[FILTER_SIZE];
    dualpix_t deriv_window[FILTER_SIZE*FILTER_SIZE]; 

	static dualpix_t deriv_lines_buffer[FILTER_SIZE][MAX_WIDTH];
	#pragma HLS ARRAY_PARTITION variable=deriv_lines_buffer complete dim=1

 // effective filtering
	L1: for(row = 0; row < height+FILTER_OFFS; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT  min=hls_MIN_H max=hls_MAX_H
		L2: for(col = 0; col < width+FILTER_OFFS; col++)
		{
			#pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT min=hls_MIN_W max=hls_MAX_W

			// Line Buffer fill
			if(col < width)
				L3:for(unsigned char ii = 0; ii < FILTER_SIZE-1; ii++)
				{
					two_pixel[ii] = deriv_lines_buffer[ii][col] = deriv_lines_buffer[ii+1][col];
				}

			//There is an offset to accomodate the active pixel region
			if((col < width) && (row < height))
			{
				two_pix(2*BITS_PER_PIXEL-1, BITS_PER_PIXEL) = inp1_img[row*MAX_WIDTH+col];
				two_pix(  BITS_PER_PIXEL-1,              0) = inp2_img[row*MAX_WIDTH+col];

				two_pixel[FILTER_SIZE-1] = deriv_lines_buffer[FILTER_SIZE-1][col] = two_pix;
			}

			//Shift right the processing window to make room for the new column
			L4:for(unsigned char ii = 0; ii < FILTER_SIZE; ii++)
				L5:for(unsigned char jj = 0; jj < FILTER_SIZE-1; jj++)
			    {
					deriv_window[ii*FILTER_SIZE+jj] = deriv_window[ii*FILTER_SIZE+jj+1];
			    }

			L6:for(unsigned char ii = 0; ii < FILTER_SIZE; ii++)
			{
				deriv_window[ii*FILTER_SIZE+FILTER_SIZE-1] = two_pixel[ii];
			}

			//This design assumes there are no edges on the boundary of the image
			if ( (row >= FILTER_OFFS) & (col >= FILTER_OFFS) &  (row < height)  & (col< width) )
			{
				  hls_derivatives_kernel(deriv_window, filt_out_x, filt_out_y, filt_out_t);
			}
			else
			{
			   filt_out_x = 0;
			   filt_out_y = 0;
			   filt_out_t = 0;
			}

			if ( (row >= FILTER_OFFS) & (col >= FILTER_OFFS) ) 
			{
				   out_Ix_img[(row-FILTER_OFFS)*MAX_WIDTH+(col-FILTER_OFFS)] = (filt_out_x);
				   out_Iy_img[(row-FILTER_OFFS)*MAX_WIDTH+(col-FILTER_OFFS)] = (filt_out_y);
				   out_It_img[(row-FILTER_OFFS)*MAX_WIDTH+(col-FILTER_OFFS)] = (filt_out_t);
			}

		} // end of L2
	} // end of L1

}


#if defined(INTEGRALS_NOT_OPTIMIZED) // O(NxN) COMPLEXITY

void hls_tyx_integration_kernel(p5sqflt_t packed_window[WINDOW_SIZE*WINDOW_SIZE],
							  sum_t &a11,  sum_t &a12, sum_t &a22, sum_t &b1, sum_t &b2)
{


	typedef int sum_t;
	// local accumulators
	sum_t sum_xx = (sum_t) 0;
	sum_t sum_xy = (sum_t) 0;
	sum_t sum_yy = (sum_t) 0;
	sum_t sum_ty = (sum_t) 0;
	sum_t sum_tx = (sum_t) 0;

	sqflt_t mult_xx, mult_xy, mult_yy, mult_tx, mult_ty;
	p5sqflt_t five_sqdata;

	unsigned short int i;
	//Compute the 2D integration
	L1:for (i = 0; i < WINDOW_SIZE*WINDOW_SIZE; i++)
	{
			five_sqdata = packed_window[i];
			mult_xx     = (sqflt_t) five_sqdata( 2*(BITS_PER_PIXEL+1)-1,                    0);
			mult_yy     = (sqflt_t) five_sqdata( 4*(BITS_PER_PIXEL+1)-1, 2*(BITS_PER_PIXEL+1));
			mult_xy     = (sqflt_t) five_sqdata( 6*(BITS_PER_PIXEL+1)-1, 4*(BITS_PER_PIXEL+1));
			mult_tx     = (sqflt_t) five_sqdata( 8*(BITS_PER_PIXEL+1)-1, 6*(BITS_PER_PIXEL+1));
			mult_ty     = (sqflt_t) five_sqdata(10*(BITS_PER_PIXEL+1)-1, 8*(BITS_PER_PIXEL+1));
			sum_xx +=  mult_xx;
			sum_xy += mult_xy;
			sum_yy +=  mult_yy;
			sum_ty += mult_ty;
			sum_tx +=  mult_tx;

	}

	a11 = sum_xx;
	a12 = sum_xy;
	a22 = sum_yy;
	 b1 = sum_tx;
	 b2 = sum_ty;

}


void hls_ComputeIntegrals(flt_t Ix_img[MAX_HEIGHT*MAX_WIDTH], flt_t  Iy_img[MAX_HEIGHT*MAX_WIDTH],  flt_t It_img[MAX_HEIGHT*MAX_WIDTH],
		                 sum_t A11_img[MAX_HEIGHT*MAX_WIDTH], sum_t A12_img[MAX_HEIGHT*MAX_WIDTH], sum_t A22_img[MAX_HEIGHT*MAX_WIDTH],
		                  sum_t B1_img[MAX_HEIGHT*MAX_WIDTH], sum_t  B2_img[MAX_HEIGHT*MAX_WIDTH], unsigned short int height, unsigned short int width)
{

	unsigned short int row, col;

	sum_t a11, a12, a22;
	sum_t b1, b2;
	flt_t x_der, y_der, t_der;

	p5sqflt_t packed5_window[WINDOW_SIZE*WINDOW_SIZE], packed5_column[WINDOW_SIZE];
	#pragma HLS ARRAY_PARTITION variable=packed5_window complete dim=1
	p5sqflt_t five_sqdata;

	p3dtyx_t packed3_column[WINDOW_SIZE];
	static p3dtyx_t packed3_lines_buffer[WINDOW_SIZE][MAX_WIDTH];
	#pragma HLS ARRAY_PARTITION variable=packed3_lines_buffer complete dim=1

	sqflt_t Ixx, Iyy, Ixy, Itx, Ity;
	p3dtyx_t three_data;


	L1: for(row = 0; row < height+WINDOW_OFFS; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT  min=hls_MIN_H max=hls_MAX_H
		L2: for(col = 0; col < width+WINDOW_OFFS; col++)
		{
			#pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT min=hls_MIN_W max=hls_MAX_W

					// Line Buffer fill
					if(col < width)
						for(unsigned char ii = 0; ii < WINDOW_SIZE-1; ii++)
						{
							packed3_column[ii] = packed3_lines_buffer[ii][col] = packed3_lines_buffer[ii+1][col];
						}

					//There is an offset to accomodate the active pixel region
					if((col < width) & (row < height))
					{
						x_der = Ix_img[row*MAX_WIDTH+col];
						y_der = Iy_img[row*MAX_WIDTH+col];
						t_der = It_img[row*MAX_WIDTH+col];

						// pack data for the lines buffer
						three_data(  (BITS_PER_PIXEL+1)-1,                    0) = x_der;
						three_data(2*(BITS_PER_PIXEL+1)-1,   (BITS_PER_PIXEL+1)) = y_der;
						three_data(3*(BITS_PER_PIXEL+1)-1, 2*(BITS_PER_PIXEL+1)) = t_der;
						packed3_column[WINDOW_SIZE-1] = packed3_lines_buffer[WINDOW_SIZE-1][col] = three_data;
					}

					// Shift right the processing window to make room for the new column
					L3:for(unsigned char ii = 0; ii < WINDOW_SIZE; ii++)
						L4:for(unsigned char jj = 0; jj < WINDOW_SIZE-1; jj++)
						{
							packed5_window[ii*WINDOW_SIZE+jj] = packed5_window[ii*WINDOW_SIZE+jj+1];
						}
					L5:for(unsigned char ii = 0; ii < WINDOW_SIZE; ii++)
					{
						#pragma HLS PIPELINE
						three_data = packed3_column[ii];
						x_der = three_data(  (BITS_PER_PIXEL+1)-1,                    0);
						y_der = three_data(2*(BITS_PER_PIXEL+1)-1,   (BITS_PER_PIXEL+1));
						t_der = three_data(3*(BITS_PER_PIXEL+1)-1, 2*(BITS_PER_PIXEL+1));

						Ixx = (sqflt_t) x_der * (sqflt_t) x_der;
						Iyy = (sqflt_t) y_der * (sqflt_t) y_der;
						Ixy = (sqflt_t) x_der * (sqflt_t) y_der;
						Itx = (sqflt_t) t_der * (sqflt_t) x_der;
						Ity = (sqflt_t) t_der * (sqflt_t) y_der;

						five_sqdata( 2*(BITS_PER_PIXEL+1)-1,                    0) = Ixx; //(17 , 0);
						five_sqdata( 4*(BITS_PER_PIXEL+1)-1, 2*(BITS_PER_PIXEL+1)) = Iyy; //(35, 18);
						five_sqdata( 6*(BITS_PER_PIXEL+1)-1, 4*(BITS_PER_PIXEL+1)) = Ixy; //(53, 36);
						five_sqdata( 8*(BITS_PER_PIXEL+1)-1, 6*(BITS_PER_PIXEL+1)) = Itx; //(71, 54);
						five_sqdata(10*(BITS_PER_PIXEL+1)-1, 8*(BITS_PER_PIXEL+1)) = Ity; //(89, 72);
						packed5_column[ii] = five_sqdata;
						packed5_window[ii*WINDOW_SIZE+WINDOW_SIZE-1] = packed5_column[ii];
					}
			//This design assumes there are no edges on the boundary of the image
			if ( (row >= WINDOW_OFFS) & (col >= WINDOW_OFFS) &  (row < height)  & (col< width) )
			{
				//Compute the 2D integration: use floating point to avoid overflow with large integration windows
				hls_tyx_integration_kernel(packed5_window, a11, a12, a22, b1, b2);
			}
			else
			{
				a11=0; a12=0; a22=0; b1=0; b2=0;
			}

			if ( (row >= WINDOW_OFFS) & (col >= WINDOW_OFFS) )
			{
				 //output data in normalized way (so that thresholding is independent on window size)
				A11_img[(row-WINDOW_OFFS)*MAX_WIDTH+(col-WINDOW_OFFS)] = a11;
				A12_img[(row-WINDOW_OFFS)*MAX_WIDTH+(col-WINDOW_OFFS)] = a12;
				A22_img[(row-WINDOW_OFFS)*MAX_WIDTH+(col-WINDOW_OFFS)] = a22;
				 B1_img[(row-WINDOW_OFFS)*MAX_WIDTH+(col-WINDOW_OFFS)]  = b1;
				 B2_img[(row-WINDOW_OFFS)*MAX_WIDTH+(col-WINDOW_OFFS)]  = b2;
			}


		} // end of L2
	} // end of L1

}

#elif defined(OPTIMIZED_TO_SAVE_DSP48) //FROM O(N) to O(1) COMPLEXITY


void hls_ComputeIntegrals(flt_t Ix_img[MAX_HEIGHT*MAX_WIDTH], flt_t  Iy_img[MAX_HEIGHT*MAX_WIDTH],  flt_t It_img[MAX_HEIGHT*MAX_WIDTH],
		                 sum_t A11_img[MAX_HEIGHT*MAX_WIDTH], sum_t A12_img[MAX_HEIGHT*MAX_WIDTH], sum_t A22_img[MAX_HEIGHT*MAX_WIDTH],
		                  sum_t B1_img[MAX_HEIGHT*MAX_WIDTH], sum_t  B2_img[MAX_HEIGHT*MAX_WIDTH],
						  unsigned short int height, unsigned short int width)
{

	unsigned short int row, col;

	sum_t a11, a12, a22;
	sum_t b1, b2;
	flt_t x_der, y_der, t_der;

	p5sqflt_t packed5_last_column, five_sqdata;

	p3dtyx_t packed3_column[WINDOW_SIZE+1];
	static p3dtyx_t packed3_lines_buffer[WINDOW_SIZE+1][MAX_WIDTH];
	#pragma HLS ARRAY_PARTITION variable=packed3_lines_buffer complete dim=1

	sqflt_t top_Ixx, top_Iyy, top_Ixy, top_Itx, top_Ity;
	sqflt_t bot_Ixx, bot_Iyy, bot_Ixy, bot_Itx, bot_Ity;
	p3dtyx_t three_data;

	static	int sum_Ixx, sum_Ixy, sum_Iyy, sum_Itx, sum_Ity; // sliding window sums. Gray color cell in Figure 18.

	// color sums for the entire image width. Yellow color cell in Figure 18
	static int csIxix [MAX_WIDTH], csIxiy [MAX_WIDTH], csIyiy [MAX_WIDTH], csDix [MAX_WIDTH], csDiy [MAX_WIDTH];
	static int cbIxix [MAX_WIDTH], cbIxiy [MAX_WIDTH], cbIyiy [MAX_WIDTH], cbDix [MAX_WIDTH], cbDiy [MAX_WIDTH];
	#pragma HLS RESOURCE   variable=csIxix core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=csIxiy core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=csIyiy core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=csDix  core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=csDiy  core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=cbIxix core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=cbIxiy core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=cbIyiy core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=cbDix  core=RAM_2P_BRAM
	#pragma HLS RESOURCE   variable=cbDiy  core=RAM_2P_BRAM
	#pragma HLS DEPENDENCE variable=csIxix inter WAR false
	#pragma HLS DEPENDENCE variable=cbIxix inter WAR false
	#pragma HLS DEPENDENCE variable=cbIxiy inter WAR false
	#pragma HLS DEPENDENCE variable=cbIyiy inter WAR false
	#pragma HLS DEPENDENCE variable=cbDix  inter WAR false
	#pragma HLS DEPENDENCE variable=cbDiy  inter WAR false
	#pragma HLS DEPENDENCE variable=csIxiy inter WAR false
	#pragma HLS DEPENDENCE variable=csIyiy inter WAR false
	#pragma HLS DEPENDENCE variable=csDix  inter WAR false
	#pragma HLS DEPENDENCE variable=csDiy  inter WAR false

	int csIxixR, csIxiyR, csIyiyR, csDixR, csDiyR; // Blue color cell in Figure 18

	// the left and right indices onto the column sums
	int zIdx =      - (WINDOW_SIZE);
	int nIdx = zIdx + (WINDOW_SIZE);

	L1: for (row = 0; row < height + WINDOW_OFFS; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT  min=hls_MIN_H max=hls_MAX_H
		L2: for (col = 0; col < width + WINDOW_OFFS; col++)
		{
		#pragma HLS PIPELINE
		#pragma HLS LOOP_TRIPCOUNT min=hls_MIN_W max=hls_MAX_W

			// line-buffer fill
			if (col < width)
				for (unsigned char ii = 0; ii < WINDOW_SIZE; ii++) {
					packed3_column[ii] = packed3_lines_buffer[ii][col] = packed3_lines_buffer[ii + 1][col]; }

			if ((col < width) & (row < height))
			{
				x_der = Ix_img[row * MAX_WIDTH + col]; y_der = Iy_img[row * MAX_WIDTH + col]; t_der = It_img[row * MAX_WIDTH + col];

				// pack data for the line-buffer
				three_data((BITS_PER_PIXEL + 1) - 1, 0) = x_der;
				three_data(2 * (BITS_PER_PIXEL + 1) - 1,   (BITS_PER_PIXEL + 1)) = y_der;
				three_data(3 * (BITS_PER_PIXEL + 1) - 1, 2*(BITS_PER_PIXEL + 1)) = t_der;
				packed3_column[WINDOW_SIZE] = packed3_lines_buffer[WINDOW_SIZE][col] = three_data;
				// line-buffer done

				// the leftSums
				int csIxixL = 0, csIxiyL = 0, csIyiyL = 0, csDixL = 0, csDiyL = 0;
				if (zIdx >= 0)
				{
					csIxixL = csIxix[zIdx]; csIxiyL = csIxiy[zIdx]; csIyiyL = csIyiy[zIdx];
					csDixL = csDix[zIdx]; csDiyL = csDiy[zIdx];
				}

				// incoming column: data on the top
				three_data = packed3_column[0];
				x_der = three_data((BITS_PER_PIXEL + 1) - 1, 0);
				y_der = three_data(2 * (BITS_PER_PIXEL + 1) - 1,     (BITS_PER_PIXEL + 1));
				t_der = three_data(3 * (BITS_PER_PIXEL + 1) - 1, 2 * (BITS_PER_PIXEL + 1));
				top_Ixx = (sqflt_t) x_der * (sqflt_t) x_der; top_Iyy = (sqflt_t) y_der * (sqflt_t) y_der;
				top_Ixy = (sqflt_t) x_der * (sqflt_t) y_der; top_Itx = (sqflt_t) t_der * (sqflt_t) x_der;
				top_Ity = (sqflt_t) t_der * (sqflt_t) y_der;

				// incoming column: data on the bottom
				three_data = packed3_column[WINDOW_SIZE];
				x_der = three_data((BITS_PER_PIXEL + 1) - 1, 0);
				y_der = three_data(2 * (BITS_PER_PIXEL + 1) - 1,   (BITS_PER_PIXEL + 1));
				t_der = three_data(3 * (BITS_PER_PIXEL + 1) - 1, 2*(BITS_PER_PIXEL + 1));
				bot_Ixx = (sqflt_t) x_der * (sqflt_t) x_der; bot_Iyy = (sqflt_t) y_der * (sqflt_t) y_der;
				bot_Ixy = (sqflt_t) x_der * (sqflt_t) y_der; bot_Itx = (sqflt_t) t_der * (sqflt_t) x_der;
				bot_Ity = (sqflt_t) t_der * (sqflt_t) y_der;

				// compute rightSums incrementally
				csIxixR=cbIxix[nIdx] + bot_Ixx - top_Ixx; csIxiyR=cbIxiy[nIdx] + bot_Ixy - top_Ixy; csIyiyR=cbIyiy[nIdx] + bot_Iyy - top_Iyy;
				csDixR = cbDix[nIdx] + bot_Itx - top_Itx;  csDiyR= cbDiy[nIdx] + bot_Ity - top_Ity;
//				csIxixR=csIxix[nIdx] + bot_Ixx - top_Ixx; csIxiyR=csIxiy[nIdx] + bot_Ixy - top_Ixy; csIyiyR=csIyiy[nIdx] + bot_Iyy - top_Iyy;
//				csDixR = csDix[nIdx] + bot_Itx - top_Itx;  csDiyR= csDiy[nIdx] + bot_Ity - top_Ity;

				// sums += (rightSums - leftLums)
				sum_Ixx += (csIxixR - csIxixL); sum_Ixy += (csIxiyR - csIxiyL); sum_Iyy += (csIyiyR - csIyiyL);
				sum_Itx += (csDixR - csDixL); 	sum_Ity += (csDiyR - csDiyL);

				// outputs
				a11 = sum_Ixx; 	a12 = sum_Ixy; 	a22 = sum_Iyy;
				b1 = sum_Itx; 	b2 = sum_Ity;

				// update new rightSums: Blue color cell in State+1 goes to Yellow color cell in State+2 of Figure 18
				cbIxix[nIdx] = csIxixR; cbIxiy[nIdx] = csIxiyR;
				cbIyiy[nIdx] = csIyiyR; cbDix[nIdx] = csDixR; cbDiy[nIdx] = csDiyR;

				csIxix[nIdx] = csIxixR; csIxiy[nIdx] = csIxiyR;
				csIyiy[nIdx] = csIyiyR; csDix[nIdx] = csDixR; csDiy[nIdx] = csDiyR;

				// update index
				zIdx++; if (zIdx == width) zIdx = 0;
				nIdx++; if (nIdx == width) nIdx = 0;

			}

			if ((row < WINDOW_OFFS) & (col < WINDOW_OFFS) & (row >= height) & (col >= width)) {
				a11 = 0; a12 = 0; a22 = 0; b1 = 0; b2 = 0; }

			if ((row >= WINDOW_OFFS) & (col >= WINDOW_OFFS))
			{
				//output data are not normalized (so that thresholding will be dependent on window size)
				A11_img[(row - WINDOW_OFFS) * MAX_WIDTH + (col - WINDOW_OFFS)] = a11;
				A12_img[(row - WINDOW_OFFS) * MAX_WIDTH + (col - WINDOW_OFFS)] = a12;
				A22_img[(row - WINDOW_OFFS) * MAX_WIDTH + (col - WINDOW_OFFS)] = a22;
				 B1_img[(row - WINDOW_OFFS) * MAX_WIDTH + (col - WINDOW_OFFS)] =  b1;
				 B2_img[(row - WINDOW_OFFS) * MAX_WIDTH + (col - WINDOW_OFFS)] =  b2;
			}

		} // end of L2
	} // end of L1


}


#else //O(N) COMPLEXITY

void hls_tyx_integration_kernel(p5sqflt_t packed5_last_column,
							    sum_t &a11,  sum_t &a12, sum_t &a22, sum_t &b1, sum_t &b2)
{

	static p5sqflt_t packed5_window[WINDOW_SIZE];
	#pragma HLS ARRAY_PARTITION variable=packed5_window complete dim=1

	// local accumulators
	static sum_t sum_Ixx, sum_Ixy, sum_Iyy, sum_Ity, sum_Itx;

	sum_t sum_xx, sum_xy, sum_yy, sum_tx, sum_ty;
	p5sqflt_t five_sqdata; 

	unsigned short int i; 

	p5sqflt_t packed5_first_column;

	//Shift right the processing window to make room for the new column
	packed5_first_column = packed5_window[0];
	L0:for(unsigned char jj = 0; jj < WINDOW_SIZE-1; jj++)
	{
		packed5_window[jj] = packed5_window[jj+1];
	}
	packed5_window[WINDOW_SIZE-1] = packed5_last_column;

	//Compute the 2D integration
	//add right-most incoming column
	five_sqdata = packed5_window[WINDOW_SIZE-1];
	sum_xx   = five_sqdata.range(  W_SUM-1,        0);
	sum_yy   = five_sqdata.range(2*W_SUM-1,    W_SUM);
	sum_xy   = five_sqdata.range(3*W_SUM-1,  2*W_SUM);
	sum_tx   = five_sqdata.range(4*W_SUM-1,  3*W_SUM);
	sum_ty   = five_sqdata.range(5*W_SUM-1,  4*W_SUM);
	sum_Ixx += sum_xx;
	sum_Ixy += sum_xy;
	sum_Iyy += sum_yy;
	sum_Ity += sum_ty;
	sum_Itx += sum_tx;

	//remove older left-most column
	five_sqdata = packed5_first_column;
	sum_xx   = five_sqdata.range(  W_SUM-1,        0);
	sum_yy   = five_sqdata.range(2*W_SUM-1,    W_SUM);
	sum_xy   = five_sqdata.range(3*W_SUM-1,  2*W_SUM);
	sum_tx   = five_sqdata.range(4*W_SUM-1,  3*W_SUM);
	sum_ty   = five_sqdata.range(5*W_SUM-1,  4*W_SUM);
	sum_Ixx -= sum_xx;
	sum_Ixy -= sum_xy;
	sum_Iyy -= sum_yy;
	sum_Ity -= sum_ty;
	sum_Itx -= sum_tx;

	a11 = sum_Ixx;
	a12 = sum_Ixy;
	a22 = sum_Iyy;
	 b1 = sum_Itx;
	 b2 = sum_Ity;

}

void hls_ComputeIntegrals(flt_t Ix_img[MAX_HEIGHT*MAX_WIDTH], flt_t  Iy_img[MAX_HEIGHT*MAX_WIDTH],  flt_t It_img[MAX_HEIGHT*MAX_WIDTH],
		                 sum_t A11_img[MAX_HEIGHT*MAX_WIDTH], sum_t A12_img[MAX_HEIGHT*MAX_WIDTH], sum_t A22_img[MAX_HEIGHT*MAX_WIDTH],
		                  sum_t B1_img[MAX_HEIGHT*MAX_WIDTH], sum_t  B2_img[MAX_HEIGHT*MAX_WIDTH], unsigned short int height, unsigned short int width)
{

	//#pragma HLS ALLOCATION instances=mul limit=150 operation
	//#pragma HLS ALLOCATION instances=fmul limit=10 operation

	unsigned short int row, col;

	sum_t a11, a12, a22;
	sum_t b1, b2;
	flt_t x_der, y_der, t_der;	

	p5sqflt_t packed5_last_column, five_sqdata;

	p3dtyx_t packed3_column[WINDOW_SIZE];
	static p3dtyx_t packed3_lines_buffer[WINDOW_SIZE][MAX_WIDTH];
	#pragma HLS ARRAY_PARTITION variable=packed3_lines_buffer complete dim=1

	sqflt_t Ixx, Iyy, Ixy, Itx, Ity;
	//#pragma HLS RESOURCE variable=Ixx core=Mul_LUT
	//#pragma HLS RESOURCE variable=Iyy core=Mul_LUT
	//#pragma HLS RESOURCE variable=Ixy core=Mul_LUT //solution_new
	//#pragma HLS RESOURCE variable=Itx core=Mul_LUT
	//#pragma HLS RESOURCE variable=Ity core=Mul_LUT

	p3dtyx_t three_data;
	sum_t sum_Ixx, sum_Ixy, sum_Iyy, sum_Itx, sum_Ity;

	L1: for(row = 0; row < height+WINDOW_OFFS; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT  min=hls_MIN_H max=hls_MAX_H
		L2: for(col = 0; col < width+WINDOW_OFFS; col++)
		{
			#pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT min=hls_MIN_W max=hls_MAX_W

			// Line Buffer fill
			if(col < width)
				for(unsigned char ii = 0; ii < WINDOW_SIZE-1; ii++)
				{
					packed3_column[ii] = packed3_lines_buffer[ii][col] = packed3_lines_buffer[ii+1][col];
				}

			// There is an offset to accomodate the active pixel region
			if((col < width) & (row < height))
			{
				x_der = Ix_img[row*MAX_WIDTH+col];
				y_der = Iy_img[row*MAX_WIDTH+col];
				t_der = It_img[row*MAX_WIDTH+col];

				// pack data for the lines buffer
				three_data(  (BITS_PER_PIXEL+1)-1,                    0) = x_der;
				three_data(2*(BITS_PER_PIXEL+1)-1,   (BITS_PER_PIXEL+1)) = y_der;
				three_data(3*(BITS_PER_PIXEL+1)-1, 2*(BITS_PER_PIXEL+1)) = t_der;
				packed3_column[WINDOW_SIZE-1] = packed3_lines_buffer[WINDOW_SIZE-1][col] = three_data;
			}

			// compute the new, incoming column
			sum_Ixx=0; sum_Iyy=0; sum_Ixy=0; sum_Itx=0; sum_Ity=0; //DB: different from src6opt
			L5:for(unsigned char ii = 0; ii < WINDOW_SIZE; ii++)
			{
				#pragma HLS PIPELINE

				three_data = packed3_column[ii];
				x_der = three_data(  (BITS_PER_PIXEL+1)-1,                    0);
				y_der = three_data(2*(BITS_PER_PIXEL+1)-1,   (BITS_PER_PIXEL+1));
				t_der = three_data(3*(BITS_PER_PIXEL+1)-1, 2*(BITS_PER_PIXEL+1));

				Ixx = (sqflt_t) x_der * (sqflt_t) x_der;
				Iyy = (sqflt_t) y_der * (sqflt_t) y_der;
				Ixy = (sqflt_t) x_der * (sqflt_t) y_der;
				Itx = (sqflt_t) t_der * (sqflt_t) x_der;
				Ity = (sqflt_t) t_der * (sqflt_t) y_der;

				sum_Ixx += Ixx;
				sum_Iyy += Iyy;
				sum_Ixy += Ixy;
				sum_Itx += Itx;
				sum_Ity += Ity;

			}

			five_sqdata.range(  W_SUM-1,        0) = sum_Ixx;
			five_sqdata.range(2*W_SUM-1,    W_SUM) = sum_Iyy;
			five_sqdata.range(3*W_SUM-1,  2*W_SUM) = sum_Ixy;
			five_sqdata.range(4*W_SUM-1,  3*W_SUM) = sum_Itx;
			five_sqdata.range(5*W_SUM-1,  4*W_SUM) = sum_Ity;

			packed5_last_column = five_sqdata;

			hls_tyx_integration_kernel(packed5_last_column, a11, a12, a22, b1, b2);

			if  ( (row < WINDOW_OFFS)&(col < WINDOW_OFFS)&(row >= height)&(col>= width) )
			{
				a11=0; a12=0; a22=0; b1=0; b2=0;
			}

			if ( (row >= WINDOW_OFFS) & (col >= WINDOW_OFFS) ) 
			{
				//output data are not normalized (so that thresholding will be dependent on window size)
				A11_img[(row-WINDOW_OFFS) * MAX_WIDTH + (col-WINDOW_OFFS)] = a11;
				A12_img[(row-WINDOW_OFFS) * MAX_WIDTH + (col-WINDOW_OFFS)] = a12;
				A22_img[(row-WINDOW_OFFS) * MAX_WIDTH + (col-WINDOW_OFFS)] = a22;
				 B1_img[(row-WINDOW_OFFS) * MAX_WIDTH + (col-WINDOW_OFFS)] =  b1;
				 B2_img[(row-WINDOW_OFFS) * MAX_WIDTH + (col-WINDOW_OFFS)] =  b2;
			}


		} // end of L2
	} // end of L1

}
#endif

#ifdef __SYNTHESIS__
int hls_ComputeVectors(sum_t A11_img[MAX_HEIGHT*MAX_WIDTH],             sum_t A12_img[MAX_HEIGHT*MAX_WIDTH],
		               sum_t A22_img[MAX_HEIGHT*MAX_WIDTH],              sum_t B1_img[MAX_HEIGHT*MAX_WIDTH],
				       sum_t  B2_img[MAX_HEIGHT*MAX_WIDTH],   
				       AXI_STREAM_S& vx_img, 
				       AXI_STREAM_S& vy_img,
					   unsigned short int height, unsigned short int width)
#else
int hls_ComputeVectors(sum_t A11_img[MAX_HEIGHT*MAX_WIDTH],             sum_t A12_img[MAX_HEIGHT*MAX_WIDTH],
		               sum_t A22_img[MAX_HEIGHT*MAX_WIDTH],              sum_t B1_img[MAX_HEIGHT*MAX_WIDTH],
				       sum_t  B2_img[MAX_HEIGHT*MAX_WIDTH],   signed short int vx_img[MAX_HEIGHT*MAX_WIDTH],
					                                               signed short int vy_img[MAX_HEIGHT*MAX_WIDTH],
					   unsigned short int height, unsigned short int width)
#endif
{

	unsigned short int row, col;
	int cnt = 0;

	float Vx, Vy;
	signed short int qVx, qVy;
	lk_union_vect_t out_vect;

	sum_t A[2][2]; 
	sum_t  B[2];

	L1: for(row = 0; row < height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT  min=hls_MIN_H max=hls_MAX_H
		L2: for(col = 0; col < width; col++)
		{
			#pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT min=hls_MIN_W max=hls_MAX_W

			Vx = 0;
			Vy = 0;

			A[0][0] = A11_img[(row)*MAX_WIDTH+(col)];	//a11
			A[0][1] = A12_img[(row)*MAX_WIDTH+(col)];   //a12;
			A[1][0] = A[0][1]; 	                        //a21
			A[1][1] = A22_img[(row)*MAX_WIDTH+(col)];   //a22;
			B[0]    =  B1_img[(row)*MAX_WIDTH+(col)];   //b1
			B[1]    =  B2_img[(row)*MAX_WIDTH+(col)];   //b2

			bool invertible = hls_matrix_inversion(A, B, THRESHOLD, Vx, Vy);
			cnt = cnt + ((int) invertible); //number of invertible points found

			////quantize motion vectors
			#ifdef __SYNTHESIS__
			PACK_S packx, packy;
			packx.data = (signed short int ) (Vx *(1<<SUBPIX_BITS));
			packx.last = (row==height-1 && col==width-1);
			vx_img.write(packx);
			packy.data = (signed short int ) (Vy *(1<<SUBPIX_BITS));
			packy.last = (row==height-1 && col==width-1);
			vy_img.write(packy);
			#else
			vx_img[(row)*MAX_WIDTH+(col)] = (signed short int ) (Vx *(1<<SUBPIX_BITS));
			vy_img[(row)*MAX_WIDTH+(col)] = (signed short int ) (Vy *(1<<SUBPIX_BITS));
			#endif

		} // end of L2

	} // end of L1

	return cnt;

}

#ifdef __SYNTHESIS__
int hls_LK(AXI_STREAM_U& inp1_img,  AXI_STREAM_U& inp2_img,
		   AXI_STREAM_S& vx_img, AXI_STREAM_S& vy_img,
		   unsigned short int height, unsigned short int width)
#else
int hls_LK(unsigned short int inp1_img[MAX_HEIGHT*MAX_WIDTH],  unsigned short int inp2_img[MAX_HEIGHT*MAX_WIDTH],
		   signed short int vx_img[MAX_HEIGHT*MAX_WIDTH], signed short int vy_img[MAX_HEIGHT*MAX_WIDTH],
		   unsigned short int height, unsigned short int width)
#endif
{

/*
#ifndef __SDSCC__
#pragma HLS INTERFACE ap_fifo port=inp1_img
#pragma HLS INTERFACE ap_fifo port=inp2_img
#pragma HLS INTERFACE ap_fifo port=vx_img
#pragma HLS INTERFACE ap_fifo port=vy_img
#endif
*/

#pragma HLS DATAFLOW

#ifdef __SYNTHESIS__ // in case of HLS-generated HW accelerator
	sum_t  A11_img[MAX_HEIGHT*MAX_WIDTH];
	sum_t  A12_img[MAX_HEIGHT*MAX_WIDTH];
	sum_t  A22_img[MAX_HEIGHT*MAX_WIDTH];
	sum_t   B1_img[MAX_HEIGHT*MAX_WIDTH];
	sum_t   B2_img[MAX_HEIGHT*MAX_WIDTH];
	flt_t  Dx1_img[MAX_HEIGHT*MAX_WIDTH]; // horizontal derivative
	flt_t  Dy1_img[MAX_HEIGHT*MAX_WIDTH]; // vertical derivative
	flt_t   Dt_img[MAX_HEIGHT*MAX_WIDTH]; // temporal derivative
	pix_t flt1_img[MAX_HEIGHT*MAX_WIDTH]; // filtered images
	pix_t flt2_img[MAX_HEIGHT*MAX_WIDTH];
	#pragma HLS STREAM variable=A11_img  depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=A12_img  depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=A22_img  depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=B1_img   depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=B2_img   depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=Dx1_img  depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=Dy1_img  depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=Dt_img   depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=flt1_img depth=HLS_STREAM_DEPTH
	#pragma HLS STREAM variable=flt2_img depth=HLS_STREAM_DEPTH
	#pragma HLS INTERFACE axis port=inp1_img
	#pragma HLS INTERFACE axis port=inp2_img
	#pragma HLS INTERFACE axis port=vx_img
	#pragma HLS INTERFACE axis port=vy_img
	#pragma HLS INTERFACE s_axilite port=height bundle=CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port=width bundle=CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS
#else // in case of purely SW execution on ARM CPU, to save stack size
	sum_t *A11_img = (sum_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(sum_t));
	sum_t *A12_img = (sum_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(sum_t));
	sum_t *A22_img = (sum_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(sum_t));
	sum_t  *B1_img = (sum_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(sum_t));
	sum_t  *B2_img = (sum_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(sum_t));
	flt_t *Dx1_img = (flt_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(flt_t));
	flt_t *Dy1_img = (flt_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(flt_t));
	flt_t  *Dt_img = (flt_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(flt_t));
	pix_t *flt1_img= (pix_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(pix_t));
	pix_t *flt2_img= (pix_t *) malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(pix_t));
#endif


	// smooth both images with same 2D filter kernel
	hls_twoIsotropicFilters(inp1_img, inp2_img, flt1_img, flt2_img, height, width);

	//compute horizontal & vertical derivatives of image 1, plus temporal derivative
	//Note: during implementation, use a 2D filter kernel with same size as the one for horiz and vert derivatives in order to have aligned Ix Iy It data
	hls_SpatialTemporalDerivatives(flt1_img, flt2_img, Dx1_img, Dy1_img, Dt_img, height, width);

	// compute integrals: remember to align the temporal derivative to the spatial ones in the implementation phase
	hls_ComputeIntegrals(Dx1_img, Dy1_img, Dt_img, A11_img, A12_img, A22_img, B1_img, B2_img, height, width);

	// compute vectors
	int cnt = hls_ComputeVectors(A11_img, A12_img, A22_img, B1_img, B2_img, vx_img, vy_img, height, width);


#ifndef __SYNTHESIS__
	free(A11_img);
	free(A12_img);
	free(A22_img);
	free(B1_img);
	free(B2_img);
	free(Dx1_img);
	free(Dy1_img);
	free(Dt_img);
	free(flt1_img);
	free(flt2_img);
#endif

	return cnt;

}
