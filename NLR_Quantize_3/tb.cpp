#include"top.h"

int main(){

	srand(30);

#ifndef TEST_CASE
	float multiplier = 0.002746367361396551;
	data_t zpW = 128, zpX = 7, zpXNext = 7;
	const int inRow = 32, inCol = 32, outRow = 32, outCol = 32;
	const int inChannel=3, outChannel=32;
	const int poolWin = 1;

#else
	float multiplier = 0.0019095869502052665;
	data_t zpW = 109, zpX = 5, zpXNext = 6;
	const int inRow = 16, inCol = 16, outRow = 8, outCol = 8;
	const int inChannel=32, outChannel=64;
	const int poolWin = 2;
#endif

	const int Tr = 8, Tc = 8;
	const int kerSize = 3;
	const int stride = 1;


	int inTiles = divide_ceil(inChannel, Ti);
	int outTiles = divide_ceil(outChannel, To);

	bool isFCN = (inRow == 1 && inCol == 1)? true:false;

	char* dataMode = (char*)"file";
	data_t ***act = Init3DArray(inRow, inCol, (inChannel>WORD_LENGTH)?inChannel:WORD_LENGTH);
	data_t ****weight = Init4DArray(kerSize, outChannel, inChannel);
	data_t ***sw_result = Init3DArray(outRow, outCol, outChannel);
	data_t ***hw_result = Init3DArray(outRow, outCol, outChannel);

	// convert to hardware data format
	uint128 *hw_input = new uint128[inRow*inCol*inChannel];
	uint128 *hw_wgt = new uint128[kerSize*kerSize*outChannel*inChannel];
	uint128 *hw_output = new uint128[outRow*outCol*outChannel];

	// initialize activation
	IFMInit<inRow, inCol, inChannel>(act, dataMode);
	IFMConvert<inRow, inCol, inChannel>(hw_input, act, inTiles);
	IFMMonitor<inRow, inCol, inChannel>(act, 0);
	IFMMonitorLinear<inRow, inCol, inChannel>(hw_input, inRow, inCol, inTiles, 0);

//	// initialize weight
	WGTInit<kerSize, outChannel, inChannel>(weight, (char*)"channel", dataMode);
	WGTConvert<kerSize, outChannel, inChannel>(hw_wgt, weight, outTiles, inTiles);
	WGTMonitor<kerSize, outChannel, inChannel>(weight, 0);
	WGTMonitorLinear<kerSize, outChannel, inChannel>(hw_wgt,outTiles, inTiles, 0);

	//read software output feature map
	ReadOFMFromFile<outRow, outCol, outChannel>(sw_result);

	// hardware conv
	DoCompute(hw_input, hw_output, hw_wgt,
	 		inRow, inCol, inChannel, outChannel,
	 		Tr, Tc, kerSize, stride, poolWin,
			multiplier, zpX, zpW, zpXNext);
//	OFMMonitorLinear(hw_output, outRow, outCol, outChannel);

	OFMConvert<outChannel>(hw_result, hw_output, outRow, outCol);

//	OFMMonitor<outChannel>(hw_result, outRow, outCol);

	int err = 0;
 	for(int k = 0; k < outChannel; k++){
 		printf("================== channel = %d ===============\n", (k));
 		for(int i = 0; i < outRow; i++){
 			for(int j = 0; j < outCol; j++){
 				if(sw_result[i][j][k] != hw_result[i][j][k]){
 					err++;
 				}
 				cout << sw_result[i][j][k] << ":" << hw_result[i][j][k] << ", ";
// 				cout << sw_result[i][j][k] << ", ";
 			}
 			printf("\n");
 		}
 	}
 	printf("==================== errors = %d ===========================\n", err);

//	float a = 123.674, b = -23.111, c = 567, d = 123.4999;
////typedef ap_ufixed<8,8,AP_RND_CONV,AP_SAT> clamp_round_t;
//typedef data_t clamp_round_t;
//	cout << "a = " << (clamp_round_t)a << ", b = " << (clamp_round_t)d << ", c = " << (clamp_round_t)c << ", d = " << (clamp_round_t)d<< endl;
	return err;

}
