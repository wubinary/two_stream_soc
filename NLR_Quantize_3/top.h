#include<stdio.h>
#include<iostream>
#include<ap_int.h>
#include<ap_fixed.h>
#include<hls_stream.h>
#include<hls_math.h>
#include<stdint.h>
#include<math.h>

using namespace std;
using namespace hls;

//#define CSIM_FLAG

#define ULTRA96
#define TEST_CASE

#define DATAWIDTH 128
#define PREC 8
#define ACC_PREC 32
#define WORD_LENGTH 16

#ifdef CSIM_FLAG
#define MAX_TILE_OUT_HEIGHT 3
#define MAX_TILE_OUT_WIDTH 3

#else
#define MAX_TILE_OUT_HEIGHT 48
#define MAX_TILE_OUT_WIDTH 48
#endif

#define MAX_STRIDE 2
#define MAX_KERNEL_SIZE 7
#define MAX_TILE_IN_HEIGHT MAX_TILE_OUT_HEIGHT*MAX_STRIDE + (MAX_KERNEL_SIZE-MAX_STRIDE)
#define MAX_TILE_IN_WIDTH MAX_TILE_OUT_WIDTH*MAX_STRIDE + (MAX_KERNEL_SIZE-MAX_STRIDE)


#ifdef ULTRA96

#define To 16 // output channel
#define Ti 16 // input channel

#else
#define To 32 // output channel
#define Ti 32 // input channel
#endif


#define WORDS_PER_LINE Ti/WORD_LENGTH

#define WGT_CHUNK_NUM 1
#define HARDCODE_VAL 1

typedef ap_uint<PREC> data_t;
typedef ap_int<ACC_PREC> psum_t;

typedef ap_ufixed<8,8,AP_RND_CONV,AP_SAT> clamp_round_t;
//typedef ap_fixed<PREC,8> data_t;
//typedef ap_fixed<PREC,8> psum_t;

typedef ap_uint<DATAWIDTH> uint128;
typedef ap_uint<Ti*WORD_LENGTH> uintTi;
typedef ap_uint<To*WORD_LENGTH> uintTo;
typedef ap_uint<To*ACC_PREC> uintAcc;

/*
 * block data
 */
template<typename T, unsigned WIDTH>
struct row_t{
	T data[WIDTH];
};

template<typename T, unsigned HEIGHT, unsigned WIDTH>
struct block_t{
	T data[HEIGHT][WIDTH];
};

void DoCompute(
		uint128 *ifm,
		uint128 *ofm,
		uint128 *raw_wgt,
		int inRow, int inCol, int inChannel, int outChannel,
		int Tr, int Tc, int kerSize, int stride, int poolWin,
		float multiplier, data_t zpX, data_t zpW, data_t zpXNext);

void Convolution(uint128 *ifm,
		uint128 *ofm,
		uint128 *raw_wgt,
		int inRow, int inCol, int inChannel, int outChannel,
		int Tr, int Tc, int kerSize, int stride,
		int tileNumX, int tileNumY, int tileNumIn, int tileNumOut,
		uintTi act[MAX_TILE_IN_HEIGHT][MAX_TILE_IN_WIDTH],
		uintAcc psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH]);

void LoadActivation(
		uint128 *ifm,
		uintTi act[MAX_TILE_IN_HEIGHT][MAX_TILE_IN_WIDTH],
		int inRow, int inCol,
		int Tr, int Tc, int anchorY, int anchorX, int offset,
		int inTr, int inTc);

void LoadWeight(
		uint128 *raw_wgt,
		uintTi wgt[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][To],
		int tidIn, int tidOut, int tileNumIn,
		int kerSize);

void WriteOutput(
		uint128 *ofm,
		psum_t psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH][To],
		float multiplier, data_t zpXNext,
		int tidY, int tidX, int tidOut,
		int Tr, int Tc, int inRow, int inCol,
		int poolWin);

void WriteDRAM(
		uint128 *output, psum_t buffer[To],
		int wordOffset, float multiplier, data_t zpXNext
		);

void PoolingOutput(
		uint128 *output,
		uintAcc psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH],
		int Tr, int Tc, int outTr, int outTc);

inline
int divide_ceil(int divident, int divisor){
#pragma HLS INLINE
	if(divident < divisor){
		return 1;
	}
	return (divident%divisor == 0)?divident/divisor: (int)(divident/divisor)+1;
}
inline
int minimum(int a, int b){
	if(a >= b){
		return b;
	}
	return a;
}

inline
bool notBoundary(int i, int j, int anchorY, int anchorX , int inRow, int inCol){
	int ptrX = anchorX + j;
	int ptrY = anchorY + i;
	return ptrX >= 0 && ptrY >= 0 && ptrX < inCol && ptrY < inRow;
}

inline
data_t ***Init3DArray(int height, int width, int channel){
	data_t*** fm = new data_t**[height];
	for(int i = 0; i < height; i++){
		fm[i] = new data_t*[width];
		for(int j = 0; j < width; j++){
			fm[i][j] = new data_t[channel];
		}
	}

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			for(int k = 0; k < channel; k++){
				fm[i][j][k] = 0;
			}
		}
	}

	return fm;
}

inline
data_t ****Init4DArray(int kerSize, int outChannel, int inChannel){
	data_t**** arr = new data_t***[kerSize];
	for(int ky = 0; ky < kerSize; ky++){
		arr[ky] = new data_t**[kerSize];
		for(int kx = 0; kx < kerSize; kx++){
			arr[ky][kx] = new data_t*[outChannel];
			for(int o = 0; o < outChannel; o++){
				arr[ky][kx][o] = new data_t[inChannel];
			}
		}
	}

	for(int ky = 0; ky < kerSize; ky++){
		for(int kx = 0; kx < kerSize; kx++){
			for(int o = 0; o < outChannel; o++){
				for(int i = 0; i < inChannel; i++){
					arr[ky][kx][o][i] = 0;
				}
			}
		}
	}

	return arr;
}

/***********************************************
 *
 * Input feature map functions
 *
 ***********************************************/

template<unsigned inRow, unsigned inCol, unsigned inChannel>
void IFMInit(data_t ***ifm, char* mode){
	for(int i = 0; i < inRow; i++){
		for(int j = 0; j < inCol; j++){
			for(int k = 0; k < inChannel; k++){
				if(mode == "rand"){
#ifndef __SYNTHESIS__
					ifm[i][j][k] = rand()%256;
#endif
				}else if(mode == "order"){
					ifm[i][j][k] = i * inCol * inChannel + j * inChannel + k;
				}else{
					ifm[i][j][k] = 1;
				}
					//
			}
		}
	}

	if(mode == "file"){
		char buff[inRow*inCol*inChannel];
		data_t temp[inChannel][inRow][inCol];
		FILE *latfile;

#ifndef TEST_CASE
		sprintf(buff,"%s","ifm.dat");
#else
		sprintf(buff,"%s","ifm_1.dat");
#endif
		latfile=fopen(buff,"r");
		fread(&(temp[0][0][0]),sizeof(data_t),inRow*inCol*inChannel,latfile);
		fclose(latfile);

		for(int k = 0; k < inChannel; k++){
			for(int i = 0; i < inRow; i++){
				for(int j = 0; j < inCol; j++){
					ifm[i][j][k] = temp[k][i][j];
				}
			}
		}
	}
}

template<unsigned inRow, unsigned inCol, unsigned inChannel>
void IFMConvert(
		uint128 hw_input[inRow*inCol*inChannel],
		data_t ***ifm,
		int inTiles){
	int ifmOffset = 0;
	// write to a linear memory array
	for(int inTid = 0; inTid < inTiles; inTid++){
		for(int i = 0; i < inRow; i++){
			for(int j = 0; j < inCol; j++){
				ifmOffset = (inTid*inRow*inCol + i*inCol + j) * WORDS_PER_LINE;
				for(int wrd = 0; wrd < WORDS_PER_LINE; wrd++){
					for(int k = 0; k < WORD_LENGTH; k++){
//						printf("(k+1)*PREC = %d, k*PREC = %d \n", (k+1)*PREC, k*PREC);
						hw_input[ifmOffset + wrd].range((k+1)*PREC-1, k*PREC)
								= ifm[i][j][inTid*Ti + wrd*WORD_LENGTH + k];
					}
				}
			}
		}
	}
}

template<unsigned height, unsigned width, unsigned inChannel>
void IFMMonitor(data_t ***act, int type){

	if(type == 1){

		for(int k = 0; k < inChannel; k++){
			cout << "Channel index = " << k << endl;
			if(k > 35) break;
			for(int i = 0; i < height; i++){
				for(int j = 0; j < width; j++){
					cout << (int)act[i][j][k] << ", ";
				}
				cout << endl;
			}
			cout << endl;
		}
	}
}

template<unsigned inRow, unsigned inCol, unsigned inChannel>
void IFMMonitorLinear(
		uint128 hw_input[inRow*inCol*inChannel], int row, int col, int inTiles, int type){

	if(type == 1){
		int offset = 0;
			for(int t = 0; t < inTiles; t++){
				for(int i = 0; i < row; i++){
					cout << "Tile: " << t << ", row: " << i << endl;
					for(int j = 0; j < col; j++){
						offset = (t*row*col + i*col + j)*WORDS_PER_LINE;
						for(int wrd = 0; wrd < WORDS_PER_LINE; wrd++){
							for(int k = 0; k < WORD_LENGTH; k++){
								cout << (int)hw_input[offset + wrd].range((k+1)*PREC-1, k*PREC) << ", ";
							}
							cout << " ||| ";
						}
						cout << endl;
					}
					cout << endl;
				}
			}
	}

}

inline
void IFMMonitorTile(
		uintTi act[MAX_TILE_IN_HEIGHT][MAX_TILE_IN_WIDTH],
	int tidY, int tidX, int tidIn,
	int inTr, int inTc){

	printf("tidY = %d, tidX = %d, tidIn = %d \n", tidY, tidX, tidIn);

	for(int i = 0; i < Ti; i++){
		cout << "Channel id = " << i << endl;
		for(int y = 0; y < inTr; y++){
			for(int x = 0; x < inTc; x++){
				cout << (int)act[y][x].range((i+1)*PREC-1, i*PREC) << ", ";
			}
			cout << endl;
		}
	}

	cout << endl;
}

/***********************************************
 *
 * Weight functions
 *
 ***********************************************/

template<unsigned kerSize, unsigned outChannel, unsigned inChannel>
void WGTInit(data_t ****wgt, char* mode, char* dataMode){

	if(mode == "channel"){

		for(int k_out = 0; k_out < outChannel; k_out++){
			for(int i = 0; i < kerSize; i++){
				for(int j = 0 ; j < kerSize; j++){
					for(int k_in = 0; k_in < inChannel; k_in++){
						if(dataMode == "rand"){
#ifndef __SYNTHESIS__
							wgt[i][j][k_out][k_in] =rand()%256;
#endif
						}else if(dataMode == "order"){
							wgt[i][j][k_out][k_in] =
							k_out * kerSize * kerSize * inChannel + i * kerSize * inChannel + j * inChannel + k_in;
						}else{
							wgt[i][j][k_out][k_in] = 1;
						}
					}
				}
			}
		}	
	}else if(mode == "row"){
		for(int k_out = 0; k_out < outChannel; k_out++){
			for(int k_in = 0; k_in < inChannel; k_in++){
				for(int i = 0; i < kerSize; i++){
					for(int j = 0; j < kerSize; j++){
						wgt[i][j][k_out][k_in] = 
								k_out * inChannel * kerSize * kerSize + k_in * kerSize * kerSize + i * kerSize + j;
					}
				}
			}
		}
	}

	if(dataMode == "file"){
		char buff[kerSize*kerSize*outChannel*inChannel];
		data_t temp[outChannel][inChannel][kerSize][kerSize];
		FILE *latfile;

#ifndef TEST_CASE
		sprintf(buff,"%s","wgt.dat");
#else
		sprintf(buff,"%s","wgt_1.dat");
#endif
		latfile=fopen(buff,"r");
		fread(&(temp[0][0][0][0]),sizeof(data_t),kerSize*kerSize*outChannel*inChannel,latfile);
		fclose(latfile);

		for(int o = 0; o < outChannel; o++){
			for(int i = 0; i < inChannel; i++){
				for(int ky = 0; ky < kerSize; ky++){
					for(int kx = 0; kx < kerSize; kx++){
//						printf("o = %d, i = %d, ky = %d, kx = %d", o,i,ky,kx);
						wgt[ky][kx][o][i] = temp[o][i][ky][kx];
					}
				}
			}
		}
	}
}

template<unsigned kerSize, unsigned outChannel, unsigned inChannel>
void WGTConvert(
		uint128 hw_wgt[kerSize*kerSize*outChannel*inChannel],
	data_t**** weight,
	int outTiles, int inTiles
	){

	// TODO: currently kernel spatial major, compare with output channel major
	for(int tidOut = 0; tidOut < outTiles; tidOut++){
		for(int tidIn = 0; tidIn < inTiles; tidIn++){
			for(int o = 0; o < To; o++){
				for(int ky = 0; ky < kerSize; ky++){
					for(int kx = 0; kx < kerSize; kx++){
						for(int wrd =0; wrd < WORDS_PER_LINE; wrd++){
							for(int i = 0; i < WORD_LENGTH; i++){
								hw_wgt[tidOut*To*inTiles*kerSize*kerSize*WORDS_PER_LINE // tidOut offset
									+tidIn*To*kerSize*kerSize*WORDS_PER_LINE // tidIn offset
									+o*kerSize*kerSize*WORDS_PER_LINE // filter offset
									+ky*kerSize*WORDS_PER_LINE // filter row offset
									+kx*WORDS_PER_LINE  // filter column in a row offset
									+wrd].range((i+1)*PREC-1, i*PREC) =
								weight[ky][kx][tidOut*To + o][tidIn*Ti + wrd*WORD_LENGTH + i];
							}
						}
					}
				}
			}
		}
	}
}

template<unsigned kerSize, unsigned outChannel, unsigned inChannel>
void WGTMonitor(data_t**** weight, int type){
	
	if(type == 1){
		for(int o = 0; o < outChannel; o++){
			cout << "Output channel index = " << o << endl;
//			if(o < 32) continue;
			for(int ky = 0; ky < kerSize; ky++){
				for(int kx = 0; kx < kerSize; kx++){
					for(int i = 0; i < inChannel; i++){
						cout << (int)weight[ky][kx][o][i] << ", ";
					}
					cout << endl;
				}
			}
			cout << endl;
		}
	}else if(type == 2){
		for(int o = 0; o < outChannel; o++){
			cout << "Output channel index = " << o << endl;
		//	if(o < 32) continue;
			for(int i = 0; i < inChannel; i++){
				for(int ky = 0; ky < kerSize; ky++){
					for(int kx = 0; kx < kerSize; kx++){
						cout << (int)weight[ky][kx][o][i] << ", ";
					}
					cout << endl;
				}
				cout << endl;
			}
		}
	}
}

template<unsigned kerSize, unsigned outChannel, unsigned inChannel>
void WGTMonitorLinear(
		uint128 hw_wgt[kerSize*kerSize*outChannel*inChannel],
	int outTiles, int inTiles, int type){

	if(type == 1){
		for(int tidOut = 0; tidOut < outTiles; tidOut++){
			if(tidOut < 1) continue;
			for(int tidIn = 0; tidIn < inTiles; tidIn++){
				for(int o = 0; o < To; o++){
					printf("\n tidOut = %d, tidIn = %d, FilterID = %d\n", tidOut, tidIn, o+tidOut*To);
					for(int ky = 0; ky < kerSize; ky++){
						for(int kx = 0; kx < kerSize; kx++){
							for(int wrd = 0; wrd < WORDS_PER_LINE; wrd++){
								for(int i = 0; i < WORD_LENGTH; i++){
									cout << (int)hw_wgt[
										tidOut*To*inTiles*kerSize*kerSize*WORDS_PER_LINE // tidOut offset
										+tidIn*To*kerSize*kerSize*WORDS_PER_LINE // tidIn offset
										+o*kerSize*kerSize*WORDS_PER_LINE // filter offset
										+ky*kerSize*WORDS_PER_LINE // filter row offset
										+kx*WORDS_PER_LINE  // filter column in a row offset
										+wrd
									].range((i+1)*PREC-1, i*PREC) << ", ";
								}
								cout << " ||| ";
							}
							cout << endl;
						}
					}
				}
			}
		}
	}
}

inline
void WGTMonitorTile(
		uintTi wgt[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][To],
	int tidOut, int tidIn, int kerSize){

	for(int o = 0; o < To; o++){
		printf("tidOut = %d, tidIn = %d, FilterID = %d \n", tidOut, tidIn, o);
		for(int ky = 0; ky < kerSize; ky++){
			for(int kx = 0; kx < kerSize; kx++){
				for(int i = 0; i < Ti; i++){
					cout << (int)wgt[ky][kx][o].range((i+1)*PREC-1, i*PREC) << ", ";
				}
				cout << endl;
			}
		}
		cout << endl;
	}
}

/***********************************************
 *
 * Output feature map functions
 *
 ***********************************************/
template<unsigned outRow, unsigned outCol, unsigned outChannel>
void ReadOFMFromFile(data_t ***ofm){
	char buff[outRow*outCol*outChannel];
	data_t temp[outChannel][outRow][outCol];
	FILE *latfile;

#ifndef TEST_CASE
		sprintf(buff,"%s","ofm.dat");
#else
		sprintf(buff,"%s","ofm_1.dat");
#endif
	latfile=fopen(buff,"r");
	fread(&(temp[0][0][0]),sizeof(data_t),outRow*outCol*outChannel,latfile);
	fclose(latfile);

	for(int k = 0; k < outChannel; k++){
		for(int i = 0; i < outRow; i++){
			for(int j = 0; j < outCol; j++){
				ofm[i][j][k] = temp[k][i][j];
			}
		}
	}

}


template<unsigned outChannel>
void OFMConvert(
		data_t ***hw_result,
		uint128 *hw_output,
		int outRow, int outCol){

	int offset = 0;
	int tileNumOut = divide_ceil(outChannel, To);
	int wordPerOutLine = To/WORD_LENGTH;
	for(int tidOut = 0; tidOut < tileNumOut; tidOut++){
		for(int y = 0; y < outRow; y++){
			for(int x = 0; x < outCol; x++){
				offset = (tidOut*outRow*outCol + y*outCol + x)*wordPerOutLine;
				for(int wrd = 0; wrd < wordPerOutLine; wrd++){
					for(int i = 0; i < WORD_LENGTH; i++){
						hw_result[y][x][tidOut*To + wrd*WORD_LENGTH + i] =
						hw_output[offset + wrd].range((i+1)*PREC-1, i*PREC);
//						printf("hw_result[%d][%d][%d] = %d \n",
//								y,x,tidOut*To + wrd*WORD_LENGTH + i,
//								(int)hw_output[offset + wrd].range((i+1)*PREC-1, i*PREC));

					}
				}
			}
		}	
	}
}

template<unsigned outChannel>
void OFMMonitor(data_t ***hw_result,int outRow, int outCol){

	for(int o = 0; o < outChannel; o++){
		cout << endl;
		cout << "Output Channel Index = " << o << endl;
		for(int x = 0; x < outRow; x++){
			for(int y = 0; y < outCol; y++){
				cout << (int)hw_result[y][x][o] << ", ";
			}
			cout << endl;
		}
	}
}

inline
void OFMMonitorLinear(
		uint128 *hw_input, int outRow, int outCol, int outChannel){

	for(int i = 0; i < outRow*outCol*outChannel/WORD_LENGTH; i++){

		for(int w = 0; w < WORD_LENGTH; w++){
			cout << (int)hw_input[i].range((w+1)*PREC-1, w*PREC) << ", ";
		}
		cout << endl;
	}
}


template<typename T>
void OFMMonitorTile(
		psum_t psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH][To], int Tr, int Tc
		, int tidX, int tidY, int tidOut
){
	for(int o = 0; o < To; o++){
		printf("TidX = %d, TidY = %d, TidOut = %d, Channel index = %d \n", tidX, tidY, tidOut, o);
		for(int i = 0; i < Tr; i++){
			for(int j = 0; j < Tc; j++){
				cout << (T)psum_output[i][j][o] << ", ";
			}
			cout << endl;
		}
		cout << endl;
	}
}



/***********************************************
 *
 * Convolution
 *
 ***********************************************/

template<unsigned inRow, unsigned inCol, unsigned kerSize, unsigned outChannel, unsigned inChannel, unsigned stride>
void SWConv(
		data_t ***sw_result,
		data_t ifm[inRow][inCol][inChannel],
		data_t weight[kerSize][kerSize][outChannel][inChannel]
		){

	int outRow = divide_ceil(inRow, stride);
	int outCol = divide_ceil(inCol, stride);
	cout << "kerSize = " << -1 <<endl;
 	for(int i = 0; i < outRow; i++){
 		for(int j = 0; j < outCol; j++){
 			for(int k_out = 0; k_out < outChannel; k_out++){
 				psum_t partial_sum = 0;
 				for(int k_in = 0; k_in < inChannel; k_in++){
 					for(int k_i = -1; k_i < kerSize-1; k_i++){
 						for(int k_j = -1; k_j < kerSize-1; k_j++){
 							if((i*stride+k_i >= 0 && j*stride+k_j >= 0) && (i*stride+k_i < inRow && j*stride+k_j < inCol)){
 								partial_sum += weight[k_i+1][k_j+1][k_out][k_in]
 											* ifm[i*stride+k_i][j*stride+k_j][k_in];
 							}
 						}
 					}
 				}
 				sw_result[i][j][k_out] = partial_sum;
 //
 			}
 		}
 	}
}

inline
void SW_Pooling(data_t ***sw_result, data_t ***sw_conv_result, int row, int column, int channel, int poolWin){

	for(int z = 0; z < channel; z++){
		for(int y = 0; y < row; y+=poolWin){
			for(int x = 0; x < column; x+=poolWin){

				data_t temp = sw_conv_result[y][x][z];
				for(int i = 0; i < poolWin; i++){
					for(int j = 0; j < poolWin; j++){
						if(sw_conv_result[y+i][x+j][z] > temp){
							temp = sw_conv_result[y+i][x+j][z];
						}
					}
				}
				sw_result[y/poolWin][x/poolWin][z] = temp;
			}
		}
	}
}

template<unsigned inRow, unsigned inCol, unsigned inChannel, unsigned outChannel, unsigned kerSize>
void SW_FCN(
		data_t ***sw_result,
		data_t ***act,
		data_t weight[kerSize][kerSize][outChannel][inChannel]){

	for(int o = 0; o < outChannel; o++){
		psum_t acc = 0;
		for(int i = 0; i < inChannel; i++){
			acc += weight[0][0][o][i] * act[0][0][i];

		}
		sw_result[0][0][o] = acc;
	}
}

template<typename ACC_T>
psum_t PE(ACC_T psum,
		uintTi act,
		uintTi wgt,
		data_t zpX,
		data_t zpW,
		int enableBit){
#pragma HLS INLINE OFF

	for(int i = 0; i < Ti; i++){
#pragma HLS UNROLL
		if(i < enableBit){
//			cout << (int)((psum_t)(act.range((i+1)*PREC-1, i*PREC)) - zpX) << " * " <<
//					(int)((psum_t)(wgt.range((i+1)*PREC-1, i*PREC)) - zpW) << ", ";

			psum += ((ap_int<9>)(act.range((i+1)*PREC-1, i*PREC)) - zpX) *
					((ap_int<9>)(wgt.range((i+1)*PREC-1, i*PREC)) - zpW);
		}
	}
//	cout << endl;
	return psum;
}

template<unsigned numPE>
void HWConv(
		uintTi wgt[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][To],
		uintTi act[MAX_TILE_IN_HEIGHT][MAX_TILE_IN_WIDTH],
		psum_t psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH][To],
		data_t zpX, data_t zpW,
		int k1, int k2, int Tr, int Tc, int stride, int padding,
		int anchorY, int anchorX,
		int inRow, int inCol, int enableBit
){
#pragma HLS ALLOCATION instances=PE limit=numPE function

#pragma HLS ARRAY_PARTITION variable=wgt dim=3 complete


	CONV_LOOP_KI:for(int ki = -padding; ki < k1-padding; ki++){
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
		CONV_LOOP_KJ:for(int kj = -padding; kj < k2-padding; kj++){
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
			CONV_LOOP_TR:for(int r = 0; r < Tr; r++){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
				CONV_LOOP_TC:for(int c = 0; c < Tc; c++){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
#pragma HLS PIPELINE
					if(notBoundary(r+ki+padding, c+kj+padding, anchorY, anchorX, inRow, inCol)){

						CONV_LOOP_TO_COMP:for(int o = 0; o < To; o++){
#pragma HLS UNROLL
							psum_output[r][c][o] = PE<psum_t>(
									psum_output[r][c][o],
								act[r+ki+padding][c+kj+padding],
								wgt[ki+padding][kj+padding][o],
								zpX, zpW, enableBit
							);
						}
					}
				}
			}
		}
	}
}

template<typename T>
void InitPSUM(T psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH][To]){
	// reset psum
	for(int i = 0; i < MAX_TILE_OUT_HEIGHT; i++){
		for(int j = 0; j < MAX_TILE_OUT_WIDTH; j++){
#pragma HLS PIPELINE
			for(int o = 0; o < To; o++){
				psum_output[i][j][o] = 0;
			}
		}
	}
}


