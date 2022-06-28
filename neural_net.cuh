#include <iostream>
#include <vector>
#include <string>

#include <cudnn_version.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h>
#include "user_iface.h"
#include "layer_params.h"
#include "utils.h"
#include <map>
// ---------------------- boDNN start ----------------------
#include "cnmem.h"
// ---------------------- boDNN end ------------------------

#ifndef NEURAL_NET
#define NEURAL_NET
class NeuralNet {
public:
	void** layer_input, ** dlayer_input, ** params;
	int* layer_input_size;
	int offnum;
	int count = 0;
	int* y, * pred_y;
	 float* loss = 0;
	float softmax_eps;
	void* one_vec;
	float init_std_dev;
	 int j = 0;
	 float avg = 0;
	int st;
	std::map<int, int>dict;
	std::map<int, int>depend;
	std::vector<LayerOp> layer_type;
	int num_layers;
	cudnnHandle_t cudnn_handle;
	cublasHandle_t cublas_handle;
	curandGenerator_t curand_gen;
	double start_time, end_time;
	cudnnDataType_t data_type;
	size_t data_type_size;
	cudnnTensorFormat_t tensor_format;
	int batch_size;
	int max = 224; //batch_size为max时，无法运行
	 int pre;
	 int pre1 = 0;
	size_t init_free_bytes, free_bytes, total_bytes;
	size_t workspace_size;
	void* workspace;
	float maxcost = 0;
	int input_channels, input_h, input_w;
	int num_classes;
	int layer_to_prefetch = -2;
	float * h_loss;
	int* h_pred_y;
	cudaEvent_t start, stop;
	// lDNN
	lDNNType ldnn_type;
	lDNNConvAlgo ldnn_conv_algo;
	cudaStream_t stream_compute, stream_memory;
	int tran;
	bool pre_alloc_conv_derivative, pre_alloc_fc_derivative, pre_alloc_batch_norm_derivative;

	void** h_layer_input;
	bool* to_offload, * prefetched;

	enum OffloadType { OFFLOAD_ALL, OFFLOAD_NONE, OFFLOAD_CONV, OFFLOAD_ALTERNATE_CONV };

	 NeuralNet(std::vector<LayerSpecifier>& layers, DataType data_type, TensorFormat tensor_format,
		float softmax_eps, float init_std_dev, lDNNType ldnn_type, lDNNConvAlgo ldnn_conv_algo);

	void getLoss(void* X, int* y, double learning_rate, std::vector<float>& fwd_ldnn_lag, std::vector<float>& bwd_ldnn_lag, bool train = true, int* correct_count = NULL, float* loss = NULL);
	void getLoss(void* X, int* y, double learning_rate, bool train = true, int* correct_count = NULL, float* loss = NULL);
	void netfree(std::vector<LayerSpecifier> &layers);
	void compareOutputCorrect(int* correct_count, int* y);
	void Initial(std::vector<LayerSpecifier>& layers, int batch_size, UpdateRule update_rule, long long dropout_seed);
	void computeLoss();//float ** h_loss ,float *&loss
	void allocate(float **& h_loss);
	int findPrefetchLayer(int cur_layer);
	bool simulateNeuralNetworkMemory(lDNNConvAlgoPref algo_pref, bool hard, size_t& exp_max_consume, size_t& max_consume,bool last,OffloadType type);
	bool simulateCNMEMMemory(size_t& max_consume);
	int lDNNOptimize(size_t& exp_max_consume, size_t& max_consume);
	void setOffload(OffloadType offload_type,bool last);
	void resetPrefetched();

	// data of time
	cudaEvent_t start_compute, stop_compute;
	void getComputationTime(void* X, int* y, double learning_rate, std::vector<float>& fwd_computation_time, std::vector<float>& bwd_computation_time);
	cudaEvent_t start_transfer, stop_transfer;
	void getTransferTime(void* X, int* y, double learning_rate, std::vector<float>& fwd_transfer_time, std::vector<float>& bwd_transfer_time);
};

#endif#pragma once
