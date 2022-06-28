#include "neural_net.cuh"
#include <time.h>
#include <cstdio>
#include <string>
#include "cuda_runtime_api.h"
using namespace std;
//想做姐姐脚下的公狗
template <typename T>
__global__ void softmaxLossBackProp(int* y, T* SO, T* dSO, int batch_size, int output_size, float eps) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= batch_size)
		return;
	int cur_class = static_cast<int>(y[i]);
	dSO[i * output_size + cur_class] = -1 / (SO[i * output_size + cur_class] * batch_size + eps);
}

template <typename T>
__global__ void computeSoftmaxLoss(T* O, int* y, float* loss, int batch_size, int num_classes, float eps) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; //一维情况下计算线程的全局id
printf("kernel i :%d\n", i);
	if (i >= batch_size)
		return;
	printf("kernel loc:%d\n", i * num_classes + y[i]);
	loss[i] = -logf(O[i * num_classes + y[i]] + eps);

}

template <typename T>
__global__ void inferClass(T* O, int* pred_y, int batch_size, int num_classes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;

	T max = O[i * num_classes];
	int index = 0;
	for (int j = 1; j < num_classes; j++) {
		if (O[i * num_classes + j] > max) {
			max = O[i * num_classes + j];
			index = j;
		}
	}
	pred_y[i] = index;
}


//表明类的函数
void NeuralNet::computeLoss() {
	//	checkCudaErrors(cudaMemcpy(h_loss, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (layer_type[num_layers - 1] == SOFTMAX) {
		if (data_type == CUDNN_DATA_FLOAT)
			computeSoftmaxLoss<float> << <ceil(1.0 * batch_size / BW),BW >> > ((float*)layer_input[num_layers], this->y, loss, batch_size, num_classes, softmax_eps);
		else if (data_type == CUDNN_DATA_DOUBLE)
			computeSoftmaxLoss<double> << <ceil(1.0 * batch_size / BW), BW >> > ((double*)layer_input[num_layers], this->y, loss, batch_size, num_classes, softmax_eps);
	}//尖括号中包括四种信息，<<<块个数，线程个数，动态分配共享内存，流>>>，其中动态分配共享内存和流不是必填项。确定块个数和线程个数的一般步骤为：
	//1） 先根据GPU设备的硬件资源确定一个块内的线程个数；再根据数据大小和每个线程处理数据个数确定块个数。
	//这一句出错
	cudaError_t cudaStatus = cudaGetLastError();
	//for (int j = 0; j < batch_size; j++) {
		//printf("%f",loss[j]);
	//}
	fprintf(stderr, "4、Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	(cudaMemcpyAsync(h_loss, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));//checkCudaErrors
	//checkCudaErrors(cudaMemcpy(h_loss, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost)); 优化的地方
	//float total_loss = 0.0;
	//for (int i = 0; i < batch_size; i++)
		//total_loss += h_loss[i];
	//return total_loss / batch_size;
}

void NeuralNet::compareOutputCorrect(int* correct_count, int* y) {
	*correct_count = 0;

	if (data_type == CUDNN_DATA_FLOAT) {
		float* typecast_O = (float*)layer_input[num_layers - 1];
		inferClass<float> << <ceil(1.0 * batch_size / BW), BW >> > (typecast_O, pred_y, batch_size, num_classes);
		checkCudaErrors(cudaMemcpy(h_pred_y, pred_y, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < batch_size; i++) {
			if (h_pred_y[i] == y[i])
				*correct_count = *correct_count + 1;
		}
	}
	else if (data_type == CUDNN_DATA_DOUBLE) {
		double* typecast_O = (double*)layer_input[num_layers - 1];
		inferClass<double> << <ceil(1.0 * batch_size / BW), BW >> > (typecast_O, pred_y, batch_size, num_classes);
		checkCudaErrors(cudaMemcpy(h_pred_y, pred_y, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < batch_size; i++) {
			if (h_pred_y[i] == y[i])
				*correct_count = *correct_count + 1;
		}
	}
}

void NeuralNet::netfree(std::vector<LayerSpecifier>&layers) {
	
	for (int i = 0; i < num_layers; i++) {
		size_t input_size;
		if (layers[i].type == CONV) {  //layers保存的是每一层的信息
			ConvDescriptor* user_params = (ConvDescriptor*)layers[i].params;
			((ConvLayerParams*)params[i])->freeSpace(pre_alloc_conv_derivative);
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor* user_params = (FCDescriptor*)layers[i].params;
			((FCLayerParams*)params[i])->freeSpace( pre_alloc_fc_derivative);
		}
		else if (layers[i].type == DROPOUT) {
			DropoutDescriptor* user_params = (DropoutDescriptor*)layers[i].params;
			((DropoutLayerParams*)params[i])->freeSpace();
		}
		else if (layers[i].type == BATCHNORM) {
			BatchNormDescriptor* user_params = (BatchNormDescriptor*)layers[i].params;
			((BatchNormLayerParams*)params[i])->freeSpace(pre_alloc_batch_norm_derivative);
			
		}
		
		// ---------------------- vDNN end ------------------------
	}
//	free(layer_input);
	
	//free(dlayer_input);
	//free(h_layer_input);
	//free(to_offload);
	//free(prefetched);
	
	//checkCudaErrors(cudaFree(y));
	//checkCudaErrors(cudaFree(pred_y));
	//checkCudaErrors(cudaFree(loss));
	//checkCudaErrors(cudaFree(one_vec));
	///checkCudaErrors(cudaFreeHost(h_loss));//使用cudaMallocHost来分配固定内存
	//checkCudaErrors(cudaFreeHost(h_pred_y));
	//for (int i = 0; i < num_layers; i++) {
		//free(params[i]);
	//}
	//free(params);
}
void NeuralNet::Initial(std::vector<LayerSpecifier>& layers,int batch_size, UpdateRule update_rule, long long dropout_seed) {
	this->batch_size = batch_size;
	LayerDimension current_output_size;
	LayerDimension prev_output_size;
	for (int i = 0; i < num_layers; i++) {
		layer_type.push_back(layers[i].type);
		if (layers[i].type == CONV) {
			ConvDescriptor* user_params = (ConvDescriptor*)layers[i].params;
			params[i] = malloc(sizeof(ConvLayerParams));
			
			((ConvLayerParams*)params[i])->initializeValues(cudnn_handle, user_params, this->data_type, batch_size, this->tensor_format,
				data_type_size, current_output_size, update_rule);  //只有卷积层的initial需要分配GPU内存
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor* user_params = (FCDescriptor*)layers[i].params;
			params[i] = malloc(sizeof(FCLayerParams));
			((FCLayerParams*)params[i])->initializeValues(user_params, batch_size, this->tensor_format, this->data_type,
				current_output_size, update_rule); //初始化
		}
		else if (layers[i].type == DROPOUT) {
			DropoutDescriptor* user_params = (DropoutDescriptor*)layers[i].params;
			params[i] = malloc(sizeof(DropoutLayerParams));
			((DropoutLayerParams*)params[i])->initializeValues(cudnn_handle, user_params, this->data_type, batch_size,
				this->tensor_format, current_output_size);

		}

		else if (layers[i].type == BATCHNORM) {
			BatchNormDescriptor* user_params = (BatchNormDescriptor*)layers[i].params;
			params[i] = malloc(sizeof(BatchNormLayerParams));
			((BatchNormLayerParams*)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, batch_size,
				current_output_size, update_rule);

		}
		else if (layers[i].type == POOLING) {
			PoolingDescriptor* user_params = (PoolingDescriptor*)layers[i].params;
			params[i] = malloc(sizeof(BatchNormLayerParams));

			((PoolingLayerParams*)params[i])->initializeValues(user_params, this->data_type, this->tensor_format,
				batch_size, current_output_size);
		}

		else if (layers[i].type == ACTV) {
			ActivationDescriptor* user_params = (ActivationDescriptor*)layers[i].params;
			params[i] = malloc(sizeof(ActivationLayerParams));
			((ActivationLayerParams*)params[i])->initializeValues(user_params, this->data_type, this->tensor_format,
				batch_size, current_output_size);
		}

		else if (layers[i].type == SOFTMAX) {
			SoftmaxDescriptor* user_params = (SoftmaxDescriptor*)layers[i].params;
			params[i] = malloc(sizeof(SoftmaxLayerParams));
			((SoftmaxLayerParams*)params[i])->initializeValues(user_params, this->data_type, this->tensor_format,
				batch_size, current_output_size);
			// std::cout << current_output_size.N << ' ' << current_output_size.C << current_output_size.H << current_output_size.W << std::endl;
		}
		if (i == 0) {
			prev_output_size = current_output_size;
		}
		// incomplete - have to check flatten and check exact dimension
		// else if (current_output_size.getTotalSize() != prev_output_size.getTotalSize()) {
		// 	std::cout << "Layer " << i << " output and next layer's input size mismatch\n";
		// 	exit(0);
		// }
	}

	// ---------------------- vDNN start ----------------------



	// ---------------------- vDNN end ------------------------
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes)); //获取空闲内存和内存总数  
	std::cout << "Free bytes just before allocate space: " << free_bytes << std::endl;  //这里往下没有问题
	// allocate space for parameters
	// Exception BatchNorm - looks like it will take lots of space if only FC layers - space taken = size of one input
	for (int i = 0; i < num_layers; i++) {
		size_t input_size;
		if (layers[i].type == CONV) {  //layers保存的是每一层的信息
			ConvDescriptor* user_params = (ConvDescriptor*)layers[i].params;
			checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
			std::cout << "conv :Free bytes just after allocate space: " << free_bytes << std::endl;
			((ConvLayerParams*)params[i])->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev,
				free_bytes, pre_alloc_conv_derivative);
			checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
			std::cout << "conv:Free bytes just after allocate space: " << free_bytes << std::endl;
			input_size = batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = user_params->input_h;
				input_w = user_params->input_w;
			}
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor* user_params = (FCDescriptor*)layers[i].params;
			checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
			std::cout << "FC:Free bytes just after allocate space: " << free_bytes << std::endl;
			((FCLayerParams*)params[i])->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev,
				free_bytes, pre_alloc_fc_derivative);
			checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
			std::cout << "FC:Free bytes just after allocate space: " << free_bytes << std::endl;
			input_size = batch_size * user_params->input_channels;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = 1;
				input_w = 1;
			}
		}
		else if (layers[i].type == DROPOUT) {
			DropoutDescriptor* user_params = (DropoutDescriptor*)layers[i].params;
			checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
			std::cout << "dropout:Free bytes just after allocate space: " << free_bytes << std::endl;
			((DropoutLayerParams*)params[i])->allocateSpace(free_bytes, cudnn_handle, user_params, dropout_seed);
			checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
			std::cout << "dropout:Free bytes just after allocate space: " << free_bytes << std::endl;
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == BATCHNORM) {
			BatchNormDescriptor* user_params = (BatchNormDescriptor*)layers[i].params;
			checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
			std::cout << "Free bytes just after allocate space: " << free_bytes << std::endl;
			((BatchNormLayerParams*)params[i])->allocateSpace(this->data_type, data_type_size,
				free_bytes, pre_alloc_batch_norm_derivative);
			checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
			std::cout << "Free bytes just after allocate space: " << free_bytes << std::endl;
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == POOLING) {
			PoolingDescriptor* user_params = (PoolingDescriptor*)layers[i].params;
			((PoolingLayerParams*)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = user_params->input_h;
				input_w = user_params->input_w;
			}
		}
		else if (layers[i].type == ACTV) {
			ActivationDescriptor* user_params = (ActivationDescriptor*)layers[i].params;
			((ActivationLayerParams*)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == SOFTMAX) {
			SoftmaxDescriptor* user_params = (SoftmaxDescriptor*)layers[i].params;
			((SoftmaxLayerParams*)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;

			// assuming this is last layer, allocate for next layer as well注释
			// checkCudaErrors(cudaMalloc(&layer_input[i + 1], input_size * data_type_size));
			// checkCudaErrors(cudaMalloc(&dlayer_input[i + 1], input_size * data_type_size));
			layer_input_size[i + 1] = input_size;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
			if (i == num_layers - 1) {
				num_classes = user_params->channels;
			}
		}

		// do not allocate memory initially注释
		// checkCudaErrors(cudaMalloc(&layer_input[i], input_size * data_type_size));
		// checkCudaErrors(cudaMalloc(&dlayer_input[i], input_size * data_type_size));

		// ---------------------- vDNN start ----------------------
		layer_input_size[i] = input_size;
		// ---------------------- vDNN end ------------------------
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	std::cout << "Free bytes just after allocate space: " << free_bytes << std::endl; //这之后不分配显存，尽管有cudamalloc
	// very small - could be allocated initially itself
	checkCudaErrors(cudaMalloc((void**)&y, batch_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&pred_y, batch_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&loss, batch_size * sizeof(float)));
	checkCudaErrors(cudaMalloc(&one_vec, batch_size * data_type_size));

	if (this->data_type == CUDNN_DATA_FLOAT)
		fillValue<float> << < ceil(1.0 * batch_size / BW), BW >> > ((float*)one_vec, batch_size, 1);
	else
		fillValue<double> << < ceil(1.0 * batch_size / BW), BW >> > ((double*)one_vec, batch_size, 1);

	checkCudaErrors(cudaMallocHost((void**)&h_loss, batch_size * sizeof(float)));//使用cudaMallocHost来分配固定内存
	checkCudaErrors(cudaMallocHost((void**)&h_pred_y, batch_size * sizeof(int)));
}

NeuralNet::NeuralNet(std::vector<LayerSpecifier>& layers, DataType data_type, TensorFormat tensor_format,
	 float softmax_eps, float init_std_dev, lDNNType ldnn_type, lDNNConvAlgo ldnn_conv_algo) {
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));//check内的函数都不用看，没有相应代码
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// ---------------------- vDNN start ----------------------
	checkCudaErrors(cudaStreamCreate(&stream_compute));
	checkCudaErrors(cudaStreamCreate(&stream_memory));
	this->ldnn_type = ldnn_type;
	this->ldnn_conv_algo = ldnn_conv_algo;
	// ---------------------- vDNN end ------------------------

	// create handle
	checkCUDNN(cudnnCreate(&cudnn_handle));   //消耗GPU内存，且没有释放，导致每次都额外占用内存

	checkCUDNN(cudnnSetStream(cudnn_handle, stream_compute)); //不消耗GPU内存
	
	checkCUBLAS(cublasCreate(&cublas_handle));  //消耗GPU内存，且没有释放，导致每次都额外占用内存

	checkCUBLAS(cublasSetStream(cublas_handle, stream_compute));
	
	checkCURAND(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT)); //消耗GPU内存，且没有释放，导致每次都额外占用内存
	
	checkCURAND(curandSetStream(curand_gen, stream_compute));

	
	
	std::cout << "Free bytes at start: " << free_bytes << "total_bytes at start" << total_bytes << std::endl;//在开了相同应用程序情况下，每次free_bytes居然一样
	st = free_bytes;
	pre_alloc_conv_derivative = false;
	pre_alloc_fc_derivative = false;
	pre_alloc_batch_norm_derivative = true;

	if (ldnn_type == lDNN_NONE) {
		pre_alloc_conv_derivative = true;
		pre_alloc_fc_derivative = true;
		pre_alloc_batch_norm_derivative = true;
	}

	if (data_type == DATA_FLOAT) {
		this->data_type = CUDNN_DATA_FLOAT;
		data_type_size = sizeof(float);
	}

	else if (data_type == DATA_DOUBLE) {
		this->data_type = CUDNN_DATA_DOUBLE;
		data_type_size = sizeof(double);
	}

	if (tensor_format == TENSOR_NCHW)
		this->tensor_format = CUDNN_TENSOR_NCHW;
	else if (tensor_format == TENSOR_NHWC)
		this->tensor_format = CUDNN_TENSOR_NHWC;

	
	this->softmax_eps = softmax_eps;
	this->init_std_dev = init_std_dev;
	num_layers = layers.size();
	
	
	// allocation of space for input to each layer
	layer_input = (void**)malloc((num_layers + 1) * sizeof(void*));
	layer_input_size = (int*)malloc((num_layers + 1) * sizeof(int));
	dlayer_input = (void**)malloc((num_layers + 1) * sizeof(void*));
	params = (void**)malloc(num_layers * sizeof(void*));
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes)); //获取空闲内存和内存总数  
	std::cout << "Free bytes just before allocate space: " << free_bytes << std::endl;
	
	
	//根据layer_type[i]来对每一层的参数进行初始化
	// allocate space in host memory for layers to be transferred
	h_layer_input = (void**)malloc(num_layers * sizeof(void*));
	to_offload = (bool*)malloc(num_layers * sizeof(bool));
	prefetched = (bool*)malloc(num_layers * sizeof(bool));
	checkCudaErrors(cudaEventCreate(&start_compute));
	checkCudaErrors(cudaEventCreate(&stop_compute));

	checkCudaErrors(cudaEventCreate(&start_transfer));
	checkCudaErrors(cudaEventCreate(&stop_transfer));
	
	
	

	// do not allocate workspace initially注释
	// allocate space for workspace and also keep track of algo
	// size_t cur_workspace_size;
	// workspace_size = 0;
	// for (int i = 0; i < num_layers; i++) {
	// 	if (layers[i].type == CONV) {
	// 		((ConvLayerParams *)params[i])->getWorkspaceSize(cur_workspace_size, free_bytes);
	// 		if (cur_workspace_size > workspace_size)
	// 			workspace_size = cur_workspace_size;
	// 	}
	// }

	// checkCudaErrors(cudaMalloc(&workspace, workspace_size));
	// free_bytes = free_bytes - workspace_size;
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	//free_bytes = 1024 * 1024 * 1200;
	// leave 600 MB and use the rest
	//std::cout << "Free bytes: " << free_bytes << std::endl;
	//printf("请输入希望给GPU预留的空间\n");
	int rest = 0;
	//cin >> rest ;
	free_bytes -= 1024 * 1024 * rest;
	// ---------------------- vDNN start ----------------------
	size_t exp_max_consume, max_consume;
	
	//	max_consume = 71912448;
		//exp_max_consume = 71912448;
	//std::cout << "actual_max_consume: " << max_consume << std::endl;
	//std::cout << "exp_max_consume: " << exp_max_consume << std::endl;
	//std::cout << "diff_max_consume(MB): " << (max_consume - exp_max_consume) / (1.0 * 1024 * 1024) << std::endl;
	//std::cout << "exp_free_bytes(MB): " << (free_bytes + 1024 * 1024 * 600 - exp_max_consume) / (1.0 * 1024 * 1024) << std::endl;
	//std::cout << "exp_total_consume(MB): " << (init_free_bytes - (free_bytes + 600 * 1024 * 1024 - exp_max_consume)) / (1.0 * 1024 * 1024) << std::endl;
	//std::cout << "actual_total_consume(MB): " << (init_free_bytes - (free_bytes + 600 * 1024 * 1024 - max_consume)) / (1.0 * 1024 * 1024) << std::endl;

	// ---------------------- vDNN end ------------------------


	// ---------------------- vDNN start ----------------------

	//free_bytes = max_consume;

	//cnmemDevice_t cnmem_device;
//	size_t cnmem_stream_memory_size = free_bytes;

	//cnmem_device.device = 0;
	//cnmem_device.size = cnmem_stream_memory_size;
//	cnmem_device.numStreams = 0;
	//cnmem_device.streams = NULL;
	//cnmem_device.streamSizes = NULL;

	// do not allow call to cudaMalloc
//	checkCNMEM(cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW));
	// ---------------------- vDNN end ------------------------

	// ---------------------- vDNN start ----------------------
//	for (int i = 0; i < num_layers; i++) {
	//	std::cerr << "to_offload[i] " << to_offload[i] << std::endl;
	//}

	//for (int i = 0; i < num_layers; i++) {
		// allocate pinned memory in host
	//	if (to_offload[i])
//			checkCudaErrors(cudaMallocHost(&h_layer_input[i], layer_input_size[i] * data_type_size));
//	}
	// ---------------------- vDNN end ------------------------
	checkCudaErrors(cudaDeviceSynchronize());
	size_t temp_free_bytes;
	checkCudaErrors(cudaMemGetInfo(&temp_free_bytes, &total_bytes));
	std::cout << "Free bytes just before end of NeuralNet: " << temp_free_bytes << std::endl;
	// {
	// 	int n;
	// 	std::cout << "waiting..\n";
	// 	std::cin >> n;
	// }

	// data of time
	
	
}
//评判在该vdnn_type下内存是否够用，
bool NeuralNet::simulateNeuralNetworkMemory(lDNNConvAlgoPref algo_pref, bool hard, size_t& exp_max_consume, size_t& max_consume,bool last, OffloadType type) {
	setOffload(type,last);  //根据offload_type设置哪些层要卸载,还有参数last决定是否卸载除ACTV、SOFTMAX外的最后一层
	printf("offnum:%d", offnum);
	cudaMemGetInfo(&free_bytes, &total_bytes);
	CnmemSpace space_tracker(free_bytes);
	space_tracker.initial_free_bytes = st;  //这个要加上
	printf("space_tracker:%d", free_bytes);
	
	std::cout << "here\n";
	std::cout << "Free bytes: " << free_bytes << std::endl;
	space_tracker.updateSpace(CnmemSpace::SUB, 0 * 1024 * 1024);
	//预取层算法优化
	int off = 0;
	for (int i = num_layers - 1; i >= 0; i--) {
		if (i > 0) {
			dict[i] = -1;
			if (layer_type[i] == ACTV || layer_type[i] == SOFTMAX ) {//|| off == offnum
				continue;
			}
			else {
				if (layer_to_prefetch == -2 || layer_to_prefetch == -1)
					layer_to_prefetch = findPrefetchLayer(i);
				if (layer_to_prefetch != -1 && i - layer_to_prefetch < 3) {
					if (layer_type[i] == POOLING || (layer_type[i] != POOLING && i - layer_to_prefetch < 2) || layer_type[i - 1] == POOLING && i - layer_to_prefetch == 2) {
						prefetched[layer_to_prefetch] = true;
						dict[i] = layer_to_prefetch;
						off++;
						printf("layer_to_prefetch:%d\n", layer_to_prefetch);
						layer_to_prefetch = -2;
					}
				}

			}
		}
	}
	//for (int i = num_layers - 1; i >= 0; i--) {
	//	printf("when i j to prefetch%d\n", dict[i]);
	//}
	max_consume = 0;
	// forward pass
	// allocate space for 1st input
	std::cerr << "Initial Used space(MB): " << space_tracker.getConsumed()/1024 /1024 << std::endl;


	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[0] * data_type_size);
	
	if (space_tracker.free_bytes <= 0){ //改进
		return false;
	}
	space_tracker.updateMaxConsume(max_consume);

	//std::cerr << "Used space after allocating input(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;

	std::cerr << "Forward pass" << std::endl;
	//判断前向传播是否内存够用
	for (int i = 0; i < num_layers; i++) {
		if (layer_type[i] == SOFTMAX)
			break;

		std::cerr << "Processing layer " << i << std::endl;

		std::cerr << "Initial Used space(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
		space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i + 1] * data_type_size);//不管哪一层，都要为下一层分配内存，因为输出要放到下一层
		if (space_tracker.free_bytes <= 0) { //改进
			return false;
		}
		std::cerr << "Used space after output allocation(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
		space_tracker.updateMaxConsume(max_consume);
		//如果是卷积层，还要额外分配工作区内存
		if (layer_type[i] == CONV) {
			ConvLayerParams* cur_params = (ConvLayerParams*)params[i];
			size_t cur_workspace_size;
			printf("%d", space_tracker.free_bytes);
			if (cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::FWD, algo_pref, hard, cur_workspace_size) != WORKSPACE_STATUS_SUCCESS) {
				return false;
			}
			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			//分配工作区，cudnn卷积计算在工作区进行
			if (space_tracker.free_bytes <= 0) { //改进
				return false;
			}
			space_tracker.updateMaxConsume(max_consume);

			if (!space_tracker.isAvailable())
				return false;
			std::cerr << "Used space after workspace allocation(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;

			// current layer computation over, deallocate workspace
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			std::cerr << "Used space after workspace deallocation(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
		}

		if (!space_tracker.isAvailable())
			return false;
		// deallocate layer input
		if (to_offload[i]) {
			std::cerr << "deallocating input to " << i << std::endl;
			//释放当前层的数据
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
			std::cerr << "Used space after deallocating input(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
		}
	}
	//前向传播判断结束
	std::cerr << "Backward pass" << std::endl;
	if (batch_size * num_classes * data_type_size != layer_input_size[num_layers] * data_type_size) {
		std::cout << "Panic!! Using wrong size\n";
		//return false;
	}

	// backward pass
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[num_layers] * data_type_size);
	if (space_tracker.free_bytes <= 0) { //改进
		return false;
	}
	std::cerr << "Used space after allocating final derivative(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
	space_tracker.updateMaxConsume(max_consume);
	// std::cerr << "max_consume: " << max_consume << std::endl;
	for (int i = num_layers - 1; i >= 0; i--) {
		// allocate space for previous layer derivative
		std::cerr << "Processing layer " << i << std::endl;
		std::cerr << "Used space initial(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
	//存疑
		if (i > 0) {
			if (layer_type[i] == SOFTMAX)
				continue;
			else {
				
				space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i] * data_type_size);
				if (space_tracker.free_bytes <= 0) { //改进
					return false;
				}
				std::cerr << "Used space after allocating prev. derivative(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
				space_tracker.updateMaxConsume(max_consume);
			}
			// std::cerr << "max_consume: " << max_consume << std::endl;
		}
		
		layer_to_prefetch = dict[i];
		// if layer to be prefetched, allocate space for that layer
		//不管是哪一层，都要为预取的层分配空间
		if (layer_to_prefetch != -1 ) {
			
			std::cerr << "Prefetch layer " << layer_to_prefetch << std::endl;
			space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[layer_to_prefetch] * data_type_size);
			if (space_tracker.free_bytes <= 0) { //改进
				return false;
			}
			
			std::cerr << "Used space after allocating prefetch(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
			space_tracker.updateMaxConsume(max_consume);
			
		}
		//卷积层除了分配权重内存，还要额外分配工作区内存
		if (layer_type[i] == CONV) {
			ConvLayerParams* cur_params = (ConvLayerParams*)params[i];
			size_t cur_filter_workspace_size;
			if (cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::BWD_FILTER, algo_pref, hard, cur_filter_workspace_size) != WORKSPACE_STATUS_SUCCESS) {
				return false;
			}
			size_t cur_data_workspace_size = 0;
			if (i > 0)
				if (cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::BWD_DATA, algo_pref, hard, cur_data_workspace_size) != WORKSPACE_STATUS_SUCCESS) {
					return false;
				}

			size_t cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;

			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			if (space_tracker.free_bytes <= 0) { //改进
				return false;
			}
			std::cerr << "Used space after allocating workspace(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
			space_tracker.updateMaxConsume(max_consume);

			if (!pre_alloc_conv_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->kernel_size * data_type_size);
				if (space_tracker.free_bytes <= 0) { //改进
					return false;
				}
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
				if (space_tracker.free_bytes <= 0) { //改进
					return false;
				}
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
			}

			// std::cerr << "max_consume: " << max_consume << std::endl;
			if (!space_tracker.isAvailable())
				return false;

			// current layer computation over, deallocate workspace
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			std::cerr << "Used space after deallocating workspace(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;

			if (!pre_alloc_conv_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
			}//优化：对权重进行卸载
		}
		//卷积层之外的层只涉及权重的预取卸载
		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams* cur_params = (FCLayerParams*)params[i];

			if (!pre_alloc_fc_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->weight_matrix_size * data_type_size);
				if (space_tracker.free_bytes <= 0) { //改进
					return false;
				}
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
				if (space_tracker.free_bytes <= 0) { //改进
					return false;
				}
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
			}

			if (!space_tracker.isAvailable())
				return false;

			if (!pre_alloc_fc_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
			}
		}
		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams* cur_params = (BatchNormLayerParams*)params[i];
			if (!pre_alloc_batch_norm_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				if (space_tracker.free_bytes <= 0) { //改进
					return false;
				}
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				if (space_tracker.free_bytes <= 0) { //改进
					return false;
				}
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
			}

			if (!space_tracker.isAvailable())
				return false;

			if (!pre_alloc_batch_norm_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
			}
		}

		if (!space_tracker.isAvailable())
			return false;
		// 所有层都会释放上一层的数据
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		printf("free_bytes:%d\n", space_tracker.free_bytes);
		std::cerr << "Used space after deallocating output, derivative(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
		// if 1st layer, deallocate input layer also
		if (i == 0) {
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
			printf("free_bytes:%d\n", space_tracker.free_bytes);
			std::cerr << "Used space after deallocating input(MB): " << space_tracker.getConsumed() / 1024 / 1024 << std::endl;
		}
	}

	if (space_tracker.getConsumed() > 0)
		std::cerr << "Panic!! more free bytes\n";
	if (space_tracker.getConsumed() != 0)
		std::cerr << "Panic!! bytes not freed properly\n";
	// return true;

	exp_max_consume = max_consume;
	// check with cnmem once
	
	space_tracker.updateSpace(CnmemSpace::ADD, 0 * 1024 * 1024);
	//bool ret_val = simulateCNMEMMemory(max_consume);  //出错语句
	
	return true;
}

bool NeuralNet::simulateCNMEMMemory(size_t& max_consume) {

	size_t init_max_consume = max_consume;
	cnmemDevice_t cnmem_device;

	size_t t;
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &t));
	std::cout << "free_bytes: " << free_bytes << std::endl;
	//free_bytes -= 100 * 1024 * 1024;
	cnmem_device.device = 0;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;

	std::string cnmem_memory_state_filename;
	if (ldnn_type == lDNN_ALL) {
		if (ldnn_conv_algo == lDNN_PERFORMANCE_OPTIMAL) {
			cnmem_memory_state_filename = "cnmem_all_p.dat";
		}
		else if (ldnn_conv_algo == lDNN_MEMORY_OPTIMAL) {
			cnmem_memory_state_filename = "cnmem_all_m.dat";
		}
	}
	else if (ldnn_type == lDNN_CONV) {
		if (ldnn_conv_algo == lDNN_PERFORMANCE_OPTIMAL) {
			cnmem_memory_state_filename = "cnmem_conv_p.dat";
		}
		else if (ldnn_conv_algo == lDNN_MEMORY_OPTIMAL) {
			cnmem_memory_state_filename = "cnmem_conv_m.dat";
		}
	}
	else if (ldnn_type == lDNN_DYN) {
		cnmem_memory_state_filename = "cnmem_dyn.dat";
	}
	else {
		cnmem_memory_state_filename = "cnmem_unknown.dat";
	}
	FILE* cnmem_memory_state_fptr = fopen(cnmem_memory_state_filename.c_str(), "w");

	size_t run_count = 0;
	bool out_of_memory = false;

	while (true) {
		run_count++;
		if (max_consume >= free_bytes)
			break;
		out_of_memory = false;
		cnmem_device.size = max_consume;

		std::cerr << run_count << ' ' << max_consume << std::endl;
		if (max_consume > free_bytes)

			std::cerr << "panic!! max_consume > free_bytes\n";

		checkCNMEM(cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW));

		resetPrefetched();
		fprintf(cnmem_memory_state_fptr, "//////////////////////////////////////////////////////////////////\n");
		fprintf(cnmem_memory_state_fptr, "run_count: %lu\n", run_count);
		fprintf(cnmem_memory_state_fptr, "max_consume: %lu\n", max_consume);
		fprintf(cnmem_memory_state_fptr, "//////////////////////////////////////////////////////////////////\n");

		fprintf(cnmem_memory_state_fptr, "initial state\n");
		cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

		//checkCNMEMSim
		if (cnmemMalloc(&layer_input[0], layer_input_size[0] * data_type_size, NULL) != CNMEM_STATUS_SUCCESS) {
			return false;
		}

		fprintf(cnmem_memory_state_fptr, "after alloc. layer_input[%d] - size: %lu\n", 0, layer_input_size[0] * data_type_size);
		cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

		// forward propagate
		for (int i = 0; i < num_layers; i++) {
			size_t cur_workspace_size;
			void* cur_workspace;
			//分配下一层也就是当前层输出的内存空间
			//checkCNMEMSim
			if(cnmemMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size, NULL) != CNMEM_STATUS_SUCCESS) {
				return false;
			}

			fprintf(cnmem_memory_state_fptr, "after alloc. layer_input[%d] - size: %lu\n", i + 1, layer_input_size[i + 1] * data_type_size);
			cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

			if (layer_type[i] == CONV) {
				// std::cout << "conv\n";
				ConvLayerParams* cur_params = (ConvLayerParams*)params[i];

				cur_workspace_size = cur_params->fwd_workspace_size;
				if(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL) != CNMEM_STATUS_SUCCESS) {
					return false;
				}

				fprintf(cnmem_memory_state_fptr, "after alloc. conv. workspace - size: %lu\n", cur_workspace_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

			}

			if (layer_type[i] == CONV) {
				if(cnmemFree(cur_workspace, NULL) != CNMEM_STATUS_SUCCESS) {
					return false;
				}
				fprintf(cnmem_memory_state_fptr, "after free conv. workspace - size: %lu\n", cur_workspace_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			}

			if (to_offload[i]) {
				if(cnmemFree(layer_input[i], NULL) != CNMEM_STATUS_SUCCESS) {
					return false;
				}
				fprintf(cnmem_memory_state_fptr, "after free layer_input[%d] - size: %lu\n", i, layer_input_size[i] * data_type_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			}

			if (layer_type[i + 1] == ACTV || layer_type[i + 1] == SOFTMAX) {
				i = i + 1;
			}
		}

		if (out_of_memory) {
			checkCNMEM(cnmemFinalize());
			if (max_consume < free_bytes)
				continue;
			else
				break;
		}

		if(cnmemMalloc(&dlayer_input[num_layers], batch_size * num_classes * data_type_size, NULL) != CNMEM_STATUS_SUCCESS) {
			return false;
		}
		fprintf(cnmem_memory_state_fptr, "after alloc. dlayer_input[%d] - size: %lu\n", num_layers, layer_input_size[num_layers] * data_type_size);
		cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

		for (int i = num_layers - 1; i >= 0; i--) {
			
			// ---------------------- vDNN start ----------------------
			size_t cur_filter_workspace_size, cur_data_workspace_size, cur_workspace_size;
			void* cur_workspace;

			if (i > 0) {
				if (layer_type[i] == ACTV || layer_type[i] == SOFTMAX) {
					dlayer_input[i] = dlayer_input[i + 1];
				}
				else {
					
					 layer_to_prefetch = dict[i];
					 
					if (layer_to_prefetch != -1) {
						
if(cnmemMalloc(&layer_input[layer_to_prefetch], layer_input_size[layer_to_prefetch] * data_type_size, NULL) != CNMEM_STATUS_SUCCESS) {
	return false;
}
						fprintf(cnmem_memory_state_fptr, "after alloc. prefetch layer_input[%d] - size: %lu\n", layer_to_prefetch, layer_input_size[layer_to_prefetch] * data_type_size);
						cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
					

					}
					printf("i:%d\n", i);
					if(cnmemMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size, NULL) != CNMEM_STATUS_SUCCESS) {
						return false;
					}
					fprintf(cnmem_memory_state_fptr, "after alloc. dlayer_input[%d] - size: %lu\n", i, layer_input_size[i] * data_type_size);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

				}
			}

			if (layer_type[i] == CONV) {
				// std::cout << "here\n";
				ConvLayerParams* cur_params = (ConvLayerParams*)params[i];

				// allocate space for derivative
				if (!pre_alloc_conv_derivative) {
					if (!cur_params->cnmemAllocDerivativesCheck(data_type_size, NULL, max_consume, free_bytes, out_of_memory))
						break;

					fprintf(cnmem_memory_state_fptr, "after alloc. dW - size: %lu\n", cur_params->kernel_size * data_type_size);
					fprintf(cnmem_memory_state_fptr, "after alloc. db - size: %lu\n", cur_params->C_out * data_type_size);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}

				cur_filter_workspace_size = cur_params->bwd_filter_workspace_size;
				if (i > 0)
					cur_data_workspace_size = cur_params->bwd_data_workspace_size;
				else
					cur_data_workspace_size = 0;
				cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;
				if(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL) != CNMEM_STATUS_SUCCESS) {
					return false;
				}

				fprintf(cnmem_memory_state_fptr, "after alloc. conv. workspace - size: %lu\n", cur_workspace_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

			}

			else if (layer_type[i] == FULLY_CONNECTED) {
				FCLayerParams* cur_params = (FCLayerParams*)params[i];

				if (!pre_alloc_fc_derivative) {
					if (!cur_params->cnmemAllocDerivativesCheck(data_type_size, NULL, max_consume, free_bytes, out_of_memory))
						break;

					fprintf(cnmem_memory_state_fptr, "after alloc. dW - size: %lu\n", cur_params->weight_matrix_size * data_type_size);
					fprintf(cnmem_memory_state_fptr, "after alloc. db - size: %lu\n", cur_params->C_out * data_type_size);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}

			else if (layer_type[i] == BATCHNORM) {
				BatchNormLayerParams* cur_params = (BatchNormLayerParams*)params[i];

				if (!pre_alloc_batch_norm_derivative) {
					if (!cur_params->cnmemAllocDerivativesCheck(data_type_size, NULL, max_consume, free_bytes, out_of_memory))
						break;

					fprintf(cnmem_memory_state_fptr, "after alloc. dscale - size: %lu\n", cur_params->allocation_size * data_type_size);
					fprintf(cnmem_memory_state_fptr, "after alloc. dbias - size: %lu\n", cur_params->allocation_size * data_type_size);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}

			else if (layer_type[i] == SOFTMAX) {
				// std::cout << "compute here\n";
				SoftmaxLayerParams* cur_params = (SoftmaxLayerParams*)params[i];
				continue;
			}

			if (layer_type[i] == CONV) {
				if (cnmemFree(cur_workspace, NULL) != CNMEM_STATUS_SUCCESS) {
					return false;
				}
				fprintf(cnmem_memory_state_fptr, "after free conv. workspace - size: %lu\n", cur_workspace_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

				if (!pre_alloc_conv_derivative) {
					ConvLayerParams* cur_params = (ConvLayerParams*)params[i];
					if (cur_params->cnmemFreeDerivatives(NULL) == false) {
						return false;
					}
					fprintf(cnmem_memory_state_fptr, "after free dP - size: %lu\n", (long unsigned)0);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}

			}
			else if (layer_type[i] == FULLY_CONNECTED) {
				if (!pre_alloc_fc_derivative) {
					FCLayerParams* cur_params = (FCLayerParams*)params[i];
					if (cur_params->cnmemFreeDerivatives(NULL) == false) {
						return false;
					}
					fprintf(cnmem_memory_state_fptr, "after free dP - size: %lu\n", (long unsigned)0);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}
			else if (layer_type[i] == BATCHNORM) {
				if (!pre_alloc_batch_norm_derivative) {
					BatchNormLayerParams* cur_params = (BatchNormLayerParams*)params[i];
					cur_params->cnmemFreeDerivatives(NULL);
					fprintf(cnmem_memory_state_fptr, "after free dP - size: %lu\n", (long unsigned)0);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}

			if (cnmemFree(layer_input[i + 1], NULL) != CNMEM_STATUS_SUCCESS){
			return false;
			}
			fprintf(cnmem_memory_state_fptr, "after free layer_input[%d] - size: %lu\n", i + 1, layer_input_size[i + 1] * data_type_size);
			cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			printf("i+1:%d\n", i+1);
			if(cnmemFree(dlayer_input[i + 1], NULL) != CNMEM_STATUS_SUCCESS){
			return false;
			}
			fprintf(cnmem_memory_state_fptr, "after free dlayer_input[%d] - size: %lu\n", i + 1, layer_input_size[i + 1] * data_type_size);
			cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			if (i == 0) {
				if(cnmemFree(layer_input[i], NULL) != CNMEM_STATUS_SUCCESS){
				return false;
			}
				fprintf(cnmem_memory_state_fptr, "after free layer_input[%d] - size: %lu\n", i, layer_input_size[i] * data_type_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			}

		}

		checkCNMEM(cnmemFinalize());
		cudaError_t cudaStatus = cudaGetLastError();
		//for (int j = 0; j < batch_size; j++) {
			//printf("%f",loss[j]);
		//}
		fprintf(stderr, "2、Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));

		if (out_of_memory) {
			if (max_consume < free_bytes)
				continue;
			else
				break;
		}
		break;
	}  //while循环结束位置
	//free_bytes += 100 * 1024 * 1024;
	if (max_consume < free_bytes) {
		double exp_size = (init_max_consume + init_free_bytes - free_bytes) / (1.0 * 1024 * 1024);
		double act_size = (max_consume + init_free_bytes - free_bytes) / (1.0 * 1024 * 1024);
		fprintf(cnmem_memory_state_fptr, "expected_memory_consume: %f MB\n", exp_size);
		fprintf(cnmem_memory_state_fptr, "actual_memory_consume: %f MB\n", act_size);
	}
	else {
		fprintf(cnmem_memory_state_fptr, "out of memory\n");
	}

	fclose(cnmem_memory_state_fptr);
	if (max_consume <= free_bytes)
		return true;
	else
		return false;
}
int batchsuccss(int batch_size , int max) {
	if (batch_size * 2 < max) {
		batch_size = int(2 * batch_size);
	}
	else if (batch_size * 1.75 < max) {
		batch_size = int(1.75 * batch_size);
	}
	else if (batch_size * 1.5 < max) {
		batch_size = int(1.5 * batch_size);
	}
	else if (batch_size * 1.25 < max) {
		batch_size = int(1.25 * batch_size);
	}
	else if(batch_size < max){

		batch_size++;
	}
	
	return batch_size;
}
int batchfail(int batch_size, int pre) {
	if (pre * 2 == batch_size) {
		batch_size = int(1.75 * pre);
	}
	else if (pre * 1.75 == batch_size) {
		batch_size = int(1.5 * pre);
	}
	else if (pre * 1.5 == batch_size) {
		batch_size = int(1.25 * pre);
	}
	else if (pre * 1.25 == batch_size) {
		batch_size = pre + 1;
	}
	else {
		batch_size--;
	}
	return batch_size;
}
int batchmid(int batch_size, int max) {
	if (batch_size * 1.5 < max) {
		batch_size = int(1.5 * batch_size);
	}
	else if (batch_size * 1.25 < max) {
		batch_size = int(1.25 * batch_size);
	}
	else {
		batch_size++;
	}
	return batch_size;
}
//根据ldnn_type尝试卸载是否可行，若不可行则进行性能较差的算法尝试
int NeuralNet::lDNNOptimize(size_t& exp_max_consume, size_t& max_consume) {

	bool hard = true, soft = false;//hard为true表示可开启内存优先选项，hard为false表示可开启

	// if type is lDNN_ALL or lDNN_CONV, check if sufficient space is available
	if (ldnn_type == lDNN_ALL) {
		
		resetPrefetched();
		//先尝试性能优先，如果不行，再尝试内存优化优先
		/*
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, true, OFFLOAD_ALL) ) {
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			pre1 = 1;
			batch_size = batchsuccss(batch_size, max);
				return 1;
		}*/
		if (pre1 == 0 && simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, true, OFFLOAD_ALL) ) {//|| (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_ALL))
			ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
			pre = batch_size;
			batch_size++;
			return 2;
		}
		
			outOfMemory();
			if (batch_size < max ) {
				max = batch_size;
			}
			batch_size = batchfail(batch_size, pre);
			return 0;
	}
	else if (ldnn_type == lDNN_CONV) {
		
		resetPrefetched();
	/*	
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, true, OFFLOAD_CONV) ) {//|| simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_CONV)
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			pre1 = 1;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			
			return 1;
		}*/
		if (pre1 == 0 && simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, true, OFFLOAD_CONV) || (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_CONV))) {
			ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
			pre = batch_size;
		if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchmid(batch_size, max);
			return 1;
		}
		
			outOfMemory();
			if (batch_size < max || max == 0) {
				max = batch_size;
			}
			batch_size = batchfail(batch_size, pre);
			
			return 0;
		
	}
	else if (ldnn_type == lDNN_NONE) {
		
		resetPrefetched();
	/*
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, true, OFFLOAD_NONE)  ) {
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			pre1 = 1;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			
			return true;
		}
		
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_NONE)) {
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return true;
		}
		*/
		if (pre1 == 0 && simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, true, OFFLOAD_NONE)) { //|| (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_NONE))
			ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return true;
		}
		
		outOfMemory();
		if (batch_size < max || max == 0) {
			max = batch_size;
		}
		batch_size = batchfail(batch_size, pre);
		
		return false;
		
	}
	else if (ldnn_type == lDNN_ALTERNATE_CONV) {
		
		resetPrefetched();

		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, true, OFFLOAD_ALTERNATE_CONV)) {// || simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_ALTERNATE_CONV)
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			pre1 = 1;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return true;
		}
		/*
		if (pre1 == 0 && simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, true, OFFLOAD_ALTERNATE_CONV) ) {//|| (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_ALTERNATE_CONV))
			ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return 1;
		}
		*/
		outOfMemory();
		if (batch_size < max || max == 0) {
			max = batch_size;
		}
		batch_size = batchfail(batch_size, pre);
		return 0;
	}

	if (ldnn_type == lDNN_DYN) {

		// check for trainability
		std::cerr << "lDNN_DYN\n";

		//setOffload(NeuralNet::OFFLOAD_ALL);
		//resetPrefetched();


		// check if work with fastest algo and no offload, if so, select it and return
		
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume,true, OFFLOAD_NONE)) {//执行到这个语句出错执行不下去
			std::cerr << "Choosing GREEDY, NO OFFLOAD\n";
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return 1;
		}
		/*
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_NONE)) {
			std::cerr << "Choosing GREEDY, NO OFFLOAD\n";
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return 1;
		}*/
		if (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume,true, OFFLOAD_NONE)) {//|| simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_NONE)
			std::cerr << "Choosing PERF_OPT, NO OFFLOAD\n";
			ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return 1;
		}
		
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume,true, OFFLOAD_ALTERNATE_CONV) || simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_ALTERNATE_CONV)) {//执行到这个语句出错执行不下去
			std::cerr << "Choosing GREEDY, CONV ALTERNATE OFFLOAD\n";
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return 1;
		}
		if (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume,true, OFFLOAD_ALTERNATE_CONV) || simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_ALTERNATE_CONV)) {//执行到这个语句出错执行不下去
			std::cerr << "Choosing PERF_OPT, CONV ALTERNATE OFFLOAD\n";
			ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return true;
		}
		// check if conv offload and fastest algo works, then check if all offload and fastest algo works
		
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume,true, OFFLOAD_CONV)|| simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_CONV)) {//执行到这个语句出错执行不下去
			std::cerr << "Choosing GREEDY, CONV OFFLOAD\n";
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size = batchsuccss(batch_size, max);
			return true;
		}
		if (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume,true, OFFLOAD_CONV) || simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_CONV)) {//执行到这个语句出错执行不下去
			std::cerr << "Choosing PERF_OPT, CONV OFFLOAD\n";
			ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batchmid(batch_size, max);
			return true;
		}

		//setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume,true, OFFLOAD_ALL) || simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_ALL)) {//执行到这个语句出错执行不下去
			std::cerr << "Choosing GREEDY, ALL OFFLOAD\n";
			ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size++;
			return true;
		}
		if (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume,true, OFFLOAD_ALL)|| simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume, false, OFFLOAD_ALL)) {//执行到这个语句出错执行不下去
			std::cerr << "Choosing PERF_OPT, ALL OFFLOAD\n";
			ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
			pre = batch_size;
			if (batch_size * 1.25 >= max) {
				return 2;
			}
			batch_size++;
			return true;
		}

		if (batch_size < max || max == 0) {
			max = batch_size;
		}
		if (pre * 2 == batch_size) {
			batch_size = int(1.5 * pre);
		}
		else if (pre * 1.75 == batch_size) {
			batch_size = int(1.25 * pre);
		}
		else {
			batch_size--;
		}
		return false;

	}
	return false;

}
//根据offload_type设置哪些层要卸载
void NeuralNet::setOffload(NeuralNet::OffloadType offload_type,bool last) {
	offnum = 0;
	if (offload_type == OFFLOAD_NONE) {
		for (int i = 0; i < num_layers; i++)
			to_offload[i] = false;
	}
	else if (offload_type == OFFLOAD_CONV) {
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == CONV) {
				to_offload[i] = true;
				offnum++;
				printf("to_offoad:%d\n", i);
			}
				
			else
				to_offload[i] = false;
		}
		if (last) {
				for (int i = num_layers - 1; i >= 0; i--) {
						if (layer_type[i] == SOFTMAX || layer_type[i] == ACTV)
							;
						else {
							to_offload[i] = false;
							printf("nooffoad:%d\n", i);
							if (layer_type[i] == CONV) {
								offnum--;
							}
							
							break;
						}
		}
		}
		// set last non SOFTMAX/ACTV layer to no_offload
	
	}
	else if (offload_type == OFFLOAD_ALL) {
		for (int i = 0; i < num_layers; i++) {
			// Only SOFTMAX, CONV, POOL, FULLY_CONNECTED used so far
			if (layer_type[i] == ACTV || layer_type[i] == SOFTMAX || layer_type[i] == FULLY_CONNECTED)
				to_offload[i] = false;
			else
			{
				offnum++;
				printf("offload%d\n", i);
				to_offload[i] = true;
			}
		}
		// set last non SOFTMAX/ACTV layer to no_offload
		//if (last) {
			//for (int i = num_layers - 1; i >= 0; i--) {
			//if (layer_type[i] == SOFTMAX || layer_type[i] == ACTV)
		//		;
		//	else {
		//		to_offload[i] = false;
		//		printf("nooffload%d\n",i);
		//		offnum--;
		//		break;
		//	}
		//}

		//}
	
	}
	else if (offload_type == OFFLOAD_ALTERNATE_CONV) {
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == CONV) {
				to_offload[i] = true;
				offnum++;
			}
				
			else
				to_offload[i] = false;
		}
		// set last non SOFTMAX/ACTV layer to no_offload
		if (last) {
			for (int i = num_layers - 1; i >= 0; i--) {
			if (layer_type[i] == SOFTMAX || layer_type[i] == ACTV)
				;
			else {
				to_offload[i] = false;
				if (layer_type[i] == CONV) {
					offnum--;
				}				
				break;
			}
		}
		}
		bool toggle = false;
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == CONV) {
				if (toggle == false) {
					to_offload[i] = false;
					offnum--;
				}
					
				toggle = !toggle;

			}
		}
		 toggle = false;
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == CONV) {
				if (toggle == false) {
					to_offload[i] = false;
					offnum--;
				}

				toggle = !toggle;

			}
		}
	}
}

void NeuralNet::resetPrefetched() {   //将所有层都标记为未预取
	for (int i = 0; i < num_layers; i++)
		prefetched[i] = false;
}

void NeuralNet::getLoss(void* X, int* y, double learning_rate, bool train, int* correct_count, float* loss) {
	std::vector<float> t1, t2;
	this->getLoss(X, y, learning_rate, t1, t2, train, correct_count, loss);
}

void NeuralNet::getLoss(void* X, int* y, double learning_rate, std::vector<float>& fwd_ldnn_lag, std::vector<float>& bwd_ldnn_lag, bool train, int* correct_count, float* scalar_loss) {
	//这函数名getLoss，但实际上是一个前向传播与反向传播的过程
	CnmemSpace space_tracker(free_bytes);
	float cost;
	for (int i = 0; i < num_layers; i++)
		prefetched[i] = false;

	(cnmemMalloc(&layer_input[0], layer_input_size[0] * data_type_size, NULL));
	//cudaMalloc(&layer_input[0], (unsigned long long)layer_input_size[0] * data_type_size);
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[0] * data_type_size);
	if (count <= 12) {
		cost = (st - space_tracker.free_bytes) / 1024 / 1024;
		if (cost > this->maxcost) {
			this->maxcost = cost;
		}
	}
	printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
		checkCudaErrors(cudaMemcpy(layer_input[0], X, batch_size * input_channels * input_h * input_w * data_type_size, cudaMemcpyHostToDevice));
		//分批次传送数据
	if (train == true) {
			(cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice));//checkCudaErrors
	}
	float alpha = 1.0, beta = 0.0;
	float Salpha = 1.0, Sbeta = 0.0;
	double Dalpha = 1.0, Dbeta = 0.0;

	// forward propagate
	for (int i = 0; i < num_layers; i++) {
		if (train == false && i == num_layers - 1)
			break;
		// ---------------------- lDNN start ----------------------
		size_t cur_workspace_size;
		void* cur_workspace;

		// offload if required
		//异步复制，在GPU中计算完后释放内存（卸载）
		if (i > 0 && to_offload[i] && train == true)  //对第0层不需要复制直接卸载就行，因为可以直接从参数X复制到GPU中
			(cudaMemcpyAsync(h_layer_input[i], layer_input[i],
				layer_input_size[i] * data_type_size, cudaMemcpyDeviceToHost, stream_memory));//checkCudaErrors
		//cudaMemcpyAsync是异步，cudamemcpy是同步，异步提高效率,同步效率太低
		checkCNMEM(cnmemMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size, NULL));//为下一层数据（输出）分配空间
		//cudaMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size);
		space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i + 1] * data_type_size);//分配完空间后就减掉消耗的空间
		if (count <= 12) {
			cost = (st - space_tracker.free_bytes) / 1024 / 1024;
			if (cost > this->maxcost) {
				this->maxcost = cost;
			}
		}
		printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
		 std::cout << "after allocation Free bytes: " << space_tracker.free_bytes << std::endl;
		// ---------------------- lDNN end ------------------------
		// std::cout << "here" << i << std::endl;
		if (layer_type[i] == CONV) {
			// std::cout << "conv\n";
			ConvLayerParams* cur_params = (ConvLayerParams*)params[i];

			cur_workspace_size = cur_params->fwd_workspace_size;
			checkCNMEM(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));//分配工作区进行计算
			
			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			printf("after malloc workspace%d\n", space_tracker.free_bytes);
				//checkCudaErrors(cudaMalloc(&cur_workspace, cur_workspace_size));
			if (count <= 12) {
				cost = (st - space_tracker.free_bytes) / 1024 / 1024;
				if (cost > this->maxcost) {
					this->maxcost = cost;
				}
			}
			printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
			std::cout << "After activation,free bytes: " << space_tracker.free_bytes << std::endl;
			// computation 只有卷积层需要分配工作区
			(cudnnConvolutionForward(cudnn_handle, &alpha,
				cur_params->input_tensor, layer_input[i],
				cur_params->filter_desc, cur_params->W,
				cur_params->conv_desc, cur_params->fwd_algo,
				cur_workspace, cur_workspace_size,
				&beta,
				cur_params->output_tensor, layer_input[i + 1]));
			//张量相加checkCUDNN
			(cudnnAddTensor(cudnn_handle, &alpha,
				cur_params->bias_desc, cur_params->b,
				&alpha,
				cur_params->output_tensor, layer_input[i + 1]));

			// 如果需要激活则进行激活checkCUDNN checkCUDNN
			if (cur_params->activation_mode != ACTIVATION_NONE) {
				(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
					&alpha,
					cur_params->output_tensor, layer_input[i + 1],
					&beta,
					cur_params->output_tensor, layer_input[i + 1]));
			}

			

		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			
			FCLayerParams* cur_params = (FCLayerParams*)params[i];
			 std::cout << "FChere" << i << std::endl;

			if (data_type == CUDNN_DATA_FLOAT) {
				checkCUBLAS(cublasSgemm(cublas_handle,//矩阵相乘
					CUBLAS_OP_N, CUBLAS_OP_N,
					cur_params->C_out, batch_size, cur_params->C_in,
					&Salpha,
					(float*)cur_params->W, cur_params->C_out,
					(float*)layer_input[i], cur_params->C_in,
					&Sbeta,
					(float*)layer_input[i + 1], cur_params->C_out));
				checkCUBLAS(cublasSgemm(cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					cur_params->C_out, batch_size, 1,
					&Salpha,
					(float*)cur_params->b, cur_params->C_out,
					(float*)one_vec, 1,
					&Salpha,
					(float*)layer_input[i + 1], cur_params->C_out));
			}
			else if (data_type == CUDNN_DATA_DOUBLE) {
				checkCUBLAS(cublasDgemm(cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					cur_params->C_out, batch_size, cur_params->C_in,
					&Dalpha,
					(double*)cur_params->W, cur_params->C_out,
					(double*)layer_input[i], cur_params->C_in,
					&Dbeta,
					(double*)layer_input[i + 1], cur_params->C_out));
				checkCUBLAS(cublasDgemm(cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					cur_params->C_out, batch_size, 1,
					&Dalpha,
					(double*)cur_params->b, cur_params->C_out,
					(double*)one_vec, 1,
					&Dalpha,
					(double*)layer_input[i + 1], cur_params->C_out));
			}
			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
					&alpha,
					cur_params->output_tensor, layer_input[i + 1],
					&beta,
					cur_params->output_tensor, layer_input[i + 1]));
			}
			 std::cout << "FChere" << i << std::endl;
		}
		else if (layer_type[i] == DROPOUT) {
			 std::cout << "Dropout\n";
			DropoutLayerParams* cur_params = (DropoutLayerParams*)params[i];
			checkCUDNN(cudnnDropoutForward(cudnn_handle, cur_params->dropout_desc,
				cur_params->input_tensor, layer_input[i],
				cur_params->input_tensor, layer_input[i + 1],
				cur_params->reserved_space,
				cur_params->reserved_space_size));
		}
		else if (layer_type[i] == BATCHNORM) {
			 std::cout << "Batchnorm\n";
			BatchNormLayerParams* cur_params = (BatchNormLayerParams*)params[i];

			if (train == true) {
				checkCUDNN(cudnnBatchNormalizationForwardTraining(cudnn_handle, cur_params->mode,
					&alpha, &beta,
					cur_params->input_tensor, layer_input[i],
					cur_params->input_tensor, layer_input[i + 1],
					cur_params->sbmv_desc,
					cur_params->scale, cur_params->bias,
					cur_params->factor,
					cur_params->running_mean, cur_params->running_variance,
					cur_params->epsilon,
					cur_params->result_save_mean, cur_params->result_save_inv_var));

			}
			else {
				checkCUDNN(cudnnBatchNormalizationForwardInference(cudnn_handle, cur_params->mode,
					&alpha, &beta,
					cur_params->input_tensor, layer_input[i],
					cur_params->input_tensor, layer_input[i + 1],
					cur_params->sbmv_desc,
					cur_params->scale, cur_params->bias,
					cur_params->running_mean, cur_params->running_variance,
					cur_params->epsilon));
			}
		}
		else if (layer_type[i] == POOLING) {
			 std::cout << "Pooling\n";
			PoolingLayerParams* cur_params = (PoolingLayerParams*)params[i];
			(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc,//checkCUDNN
				&alpha,
				cur_params->input_tensor, layer_input[i],
				&beta,
				cur_params->output_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == ACTV) {
			 std::cout << "Actv\n";
			std::cout << "Panic!! ACTV wrong place\n";
			exit(0);
			ActivationLayerParams* cur_params = (ActivationLayerParams*)params[i];
			(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,//checkCUDNN
				&alpha,
				cur_params->input_tensor, layer_input[i],
				&beta,
				cur_params->input_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == SOFTMAX) {
			 std::cout << "Softmax\n";
			std::cout << "Panic!! SOFTMAX wrong place\n";
			exit(0);
			if (train == true) {
				SoftmaxLayerParams* cur_params = (SoftmaxLayerParams*)params[i];
				checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
					&alpha,
					cur_params->input_tensor, layer_input[i],
					&beta,
					cur_params->input_tensor, layer_input[i + 1]));
			}
		}

		// ---------------------- lDNN start ----------------------
		// synchronization
		// checkCudaErrors(cudaDeviceSynchronize());注释

		// if next layer is ACTV or SOFTMAX, complete that and come to synchronization
		// the case in above if for ACTV and SOFTMAX never occurs
		if (layer_type[i + 1] == SOFTMAX) {
			i++;
			if (train == true) {
				layer_input[i + 1] = layer_input[i];
				SoftmaxLayerParams* cur_params = (SoftmaxLayerParams*)params[i];
				checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
					&alpha,
					cur_params->input_tensor, layer_input[i],
					&beta,
					cur_params->input_tensor, layer_input[i + 1]));
			}
			i--;
		}

		double start_time, end_time;
		//同步计算流，怕计算完异步复制还没结束
		//checkCudaErrors(cudaStreamSynchronize(stream_compute));

		if (train)
			start_time = clock();
		


		if (train) {
			end_time = clock();
			float lag = (end_time - start_time);
			fwd_ldnn_lag.push_back(lag);
		}
		// std::cout << "EndSynchere" << i << std::endl;
		//卷积层有工作区，需要释放内存
		if (layer_type[i] == CONV) {
			checkCNMEM(cnmemFree(cur_workspace, NULL));
			//checkCudaErrors(cudaFree(cur_workspace));
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
		}
		
		//要卸载的层释放内存
		if (to_offload[i] && train == true) {
			//同步异步复制，怕异步复制完同步计算流还没结束
			
			checkCudaErrors(cudaStreamSynchronize(stream_memory));
			
			printf("卸载的数据大小%d以及是哪一层%d\n", layer_input_size[i],layer_type[i]);

		
			checkCNMEM(cnmemFree(layer_input[i], NULL));
			//checkCudaErrors(cudaFree(layer_input[i]));
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);

		j++;
		float cost = (st - space_tracker.free_bytes) / 1024 / 1024;
		if (cost > this -> maxcost) {
			this -> maxcost = cost;
		}
		printf("某时刻占用的最大GPU内存为%f(MB)\n", this -> maxcost);
	
		}
		
		if (train == false) {
			checkCNMEM(cnmemFree(layer_input[i], NULL));
		//	checkCudaErrors(cudaFree(layer_input[i]));
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
		}
		
		if (layer_type[i + 1] == ACTV || layer_type[i + 1] == SOFTMAX) {
			i = i + 1;
		}
		
		// std::cout << "EndSynchere" << i << std::endl;

		// ---------------------- lDNN end ------------------------
	}

	// std::cout << "here" << std::endl;
	if (train == false) {
		compareOutputCorrect(correct_count, y);
		checkCNMEM(cnmemFree(layer_input[num_layers - 1], NULL));
		//checkCudaErrors(cudaFree(layer_input[num_layers - 1]));
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[num_layers - 1] * data_type_size);
		return;
	}
	  computeLoss();
	//反向传播
	// ---------------------- lDNN start ----------------------
	checkCNMEM(cnmemMalloc(&dlayer_input[num_layers], layer_input_size[num_layers] * data_type_size, NULL)); //batch_size * num_classes
	//  checkCudaErrors(cudaMalloc(&dlayer_input[num_layers], layer_input_size[num_layers] * data_type_size));
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[num_layers] * data_type_size);
	if (count <= 12) {
cost = (st - space_tracker.free_bytes) / 1024 / 1024;
	if (cost > this->maxcost) {
		this->maxcost = cost;
	}
	}
	 
	printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
	std::cout << "Free bytes: " << space_tracker.free_bytes << std::endl;
	// ---------------------- lDNN end ------------------------
	if (layer_type[num_layers - 1] == SOFTMAX) {
		// SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[num_layers - 1];注释
		if (data_type == CUDNN_DATA_FLOAT) {
			checkCudaErrors(cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(float)));
			softmaxLossBackProp<float> << <ceil(1.0 * batch_size / BW), BW >> > (this->y, (float*)layer_input[num_layers],
				(float*)dlayer_input[num_layers], batch_size, num_classes, softmax_eps);
		}
		else if (data_type == CUDNN_DATA_DOUBLE) {
			checkCudaErrors(cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(double)));
			softmaxLossBackProp<double> << <ceil(1.0 * batch_size / BW), BW >> > (this->y, (double*)layer_input[num_layers],
				(double*)dlayer_input[num_layers], batch_size, num_classes, softmax_eps);
		}
	}
	float total_loss = 0.0;     //优化的地方 将loss数据异步传输，节省了时间
	checkCudaErrors(cudaStreamSynchronize(stream_memory));
	for (int i = 0; i < batch_size; i++)
		total_loss += h_loss[i];
	*scalar_loss = total_loss;
	for (int i = num_layers - 1; i >= 0; i--) {
		// ---------------------- lDNN start ----------------------
		size_t cur_filter_workspace_size, cur_data_workspace_size, cur_workspace_size;
		void* cur_workspace;

		if (i > 0) {

			if (layer_type[i] == ACTV || layer_type[i] == SOFTMAX) {
				dlayer_input[i] = dlayer_input[i + 1];
			}
			else {
				
				layer_to_prefetch = dict[i];
				if (layer_to_prefetch != -1 ) {
					
						
					checkCNMEM(cnmemMalloc(&layer_input[layer_to_prefetch], layer_input_size[layer_to_prefetch] * data_type_size, NULL));
					//checkCudaErrors(cudaMalloc(&layer_input[layer_to_prefetch], layer_input_size[layer_to_prefetch] * data_type_size));
					space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[layer_to_prefetch] * data_type_size);
					if (count <= 12) {
						cost = (st - space_tracker.free_bytes) / 1024 / 1024;
						if (cost > this->maxcost) {
							this->maxcost = cost;
						}
					}
					printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
				 std::cout << "Free bytes: " << space_tracker.free_bytes << std::endl;
					if (layer_to_prefetch != 0) {

						//预取数据
						//checkCudaErrors(cudaEventRecord(start));//记录当前时间
					//	float milli;

						(cudaMemcpyAsync(layer_input[layer_to_prefetch], h_layer_input[layer_to_prefetch],
							layer_input_size[layer_to_prefetch] * data_type_size, cudaMemcpyHostToDevice, stream_memory));//checkCudaErrors
						checkCudaErrors(cudaStreamSynchronize(stream_memory));
					//	checkCudaErrors(cudaEventRecord(stop)); //记录当前时间
					//	checkCudaErrors(cudaEventSynchronize(stop)); //等待事件结束 
					//	checkCudaErrors(cudaEventElapsedTime(&milli, start, stop)); //计时
					//	cout << layer_to_prefetch << "层数据传输的时间为(ms)：" << milli << endl;
						cout << "预取的是什么类型的层数据" << layer_type[layer_to_prefetch] << endl;
					}
					else { //第0层数据直接从输入的数据中复制
						// std::cout << "transfer here\n";
						checkCudaErrors(cudaMemcpyAsync(layer_input[layer_to_prefetch], X,
							layer_input_size[layer_to_prefetch] * data_type_size, cudaMemcpyHostToDevice, stream_memory));
						// std::cout << "transfer here\n";
					}
				
				}
				checkCNMEM(cnmemMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size, NULL));
				//checkCudaErrors(cudaMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size));
				space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i] * data_type_size);
				if (count <= 12 ){
					cost = (st - space_tracker.free_bytes) / 1024 / 1024;
					if (cost > this->maxcost) {
						this->maxcost = cost;
					}
				}
				printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
			}
			
		}
		// ---------------------- lDNN end ------------------------
//printf("池化层代号为：%d", POOLING);
		if (layer_type[i] == CONV) {
			//
			
			// std::cout << "here\n";
		//	checkCudaErrors(cudaEventRecord(start));//记录当前时间
		//	float milli = 0.0;
			ConvLayerParams* cur_params = (ConvLayerParams*)params[i];

			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
					cur_params->output_tensor, layer_input[i + 1],
					cur_params->output_tensor, dlayer_input[i + 1],
					cur_params->output_tensor, layer_input[i + 1],
					&beta,
					cur_params->output_tensor, dlayer_input[i + 1]));
			}

			// allocate space for derivative
			if (!pre_alloc_conv_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
				if (count <= 12) {
					cost = (st - space_tracker.free_bytes) / 1024 / 1024;
					if (cost > this->maxcost) {
						this->maxcost = cost;
					}
				}
				printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
				std::cout << "here\n";
				std::cout << "Free bytes: " << space_tracker.free_bytes << std::endl;
			}

			cur_filter_workspace_size = cur_params->bwd_filter_workspace_size;
			if (i > 0)
				cur_data_workspace_size = cur_params->bwd_data_workspace_size;
			else
				cur_data_workspace_size = 0;
			// std::cout << "bwd cur_workspace_size: " << cur_workspace_size << std::endl;
			
			cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;
			
			checkCNMEM(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));
			//checkCudaErrors(cudaMalloc(&cur_workspace, cur_workspace_size));

			checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handle, &alpha,
				cur_params->output_tensor, dlayer_input[i + 1],
				&beta,
				cur_params->bias_desc, cur_params->db));

			// std::cout << "neural_net: backward conv i:" << i << std::endl;

			checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handle, &alpha,
				cur_params->input_tensor, layer_input[i],
				cur_params->output_tensor, dlayer_input[i + 1],
				cur_params->conv_desc, cur_params->bwd_filter_algo,
				cur_workspace, cur_workspace_size,
				&beta,
				cur_params->filter_desc,
				cur_params->dW));
			if (i > 0)
				checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle, &alpha,
					cur_params->filter_desc, cur_params->W,
					cur_params->output_tensor, dlayer_input[i + 1],
					cur_params->conv_desc, cur_params->bwd_data_algo,
					cur_workspace, cur_workspace_size,
					&beta,
					cur_params->input_tensor, dlayer_input[i]));

			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			if (count <= 12) {
				cost = (st - space_tracker.free_bytes) / 1024 / 1024;
				if (cost > this->maxcost) {
					this->maxcost = cost;
				}
			}
			printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
			 std::cout << "Free bytes: " << space_tracker.free_bytes << std::endl;
			// std::cout << "here\n";
			cur_params->stepParams(cublas_handle, learning_rate);
		//	checkCudaErrors(cudaEventRecord(stop)); //记录当前时间
		//	checkCudaErrors(cudaEventSynchronize(stop)); //等待事件结束 
		//	checkCudaErrors(cudaEventElapsedTime(&milli, start, stop)); //计时
		//	cout << i << "卷积层反向传播计算时间为(ms)：" <<milli << endl << endl;
		}

		else if (layer_type[i] == FULLY_CONNECTED) {
		//	checkCudaErrors(cudaEventRecord(start));//记录当前时间
		//	float milli = 0.0;
			FCLayerParams* cur_params = (FCLayerParams*)params[i];

			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
					cur_params->output_tensor, layer_input[i + 1],
					cur_params->output_tensor, dlayer_input[i + 1],
					cur_params->output_tensor, layer_input[i + 1],
					&beta,
					cur_params->output_tensor, dlayer_input[i + 1]));
			}

			if (!pre_alloc_fc_derivative) {
				
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
				if(count <= 12) {
					cost = (st - space_tracker.free_bytes) / 1024 / 1024;
					if (cost > this->maxcost) {
						this->maxcost = cost;
					}
				}
				printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
				std::cout << "here\n";
				std::cout << "Free bytes: " << space_tracker.free_bytes << std::endl;
			}

			if (data_type == CUDNN_DATA_FLOAT) {
				// bias backward
				checkCUBLAS(cublasSgemm(cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					cur_params->C_out, 1, batch_size,
					&Salpha,
					(float*)dlayer_input[i + 1], cur_params->C_out,
					(float*)one_vec, batch_size,
					&Sbeta,
					(float*)cur_params->db, cur_params->C_out));

				// weight backward
				checkCUBLAS(cublasSgemm(cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_T,
					cur_params->C_out, cur_params->C_in, batch_size,
					&Salpha,
					(float*)dlayer_input[i + 1], cur_params->C_out,
					(float*)layer_input[i], cur_params->C_in,
					&Sbeta,
					(float*)cur_params->dW, cur_params->C_out));

				// data backward
				if (i > 0)
					checkCUBLAS(cublasSgemm(cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						cur_params->C_in, batch_size, cur_params->C_out,
						&Salpha,
						(float*)cur_params->W, cur_params->C_out,
						(float*)dlayer_input[i + 1], cur_params->C_out,
						&Sbeta,
						(float*)dlayer_input[i], cur_params->C_in));
			}//

			else if (data_type == CUDNN_DATA_DOUBLE) {
				// bias backward
				checkCUBLAS(cublasDgemm(cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					cur_params->C_out, 1, batch_size,
					&Dalpha,
					(double*)dlayer_input[i + 1], cur_params->C_out,
					(double*)one_vec, batch_size,
					&Dbeta,
					(double*)cur_params->db, cur_params->C_out));

				// weight backward
				checkCUBLAS(cublasDgemm(cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_T,
					cur_params->C_out, cur_params->C_in, batch_size,
					&Dalpha,
					(double*)dlayer_input[i + 1], cur_params->C_out,
					(double*)layer_input[i], cur_params->C_in,
					&Dbeta,
					(double*)cur_params->dW, cur_params->C_out));

				// data backward
				if (i > 0)
					checkCUBLAS(cublasDgemm(cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						cur_params->C_in, batch_size, cur_params->C_out,
						&Dalpha,
						(double*)cur_params->W, cur_params->C_out,
						(double*)dlayer_input[i + 1], cur_params->C_out,
						&Dbeta,
						(double*)dlayer_input[i], cur_params->C_in));
			}
			cur_params->stepParams(cublas_handle, learning_rate);
		//	checkCudaErrors(cudaEventRecord(stop)); //记录当前时间
		//	checkCudaErrors(cudaEventSynchronize(stop)); //等待事件结束 
		//	checkCudaErrors(cudaEventElapsedTime(&milli, start, stop)); //计时
		//	cout << i << "全连接层反向传播的时间为(ms)：" << milli << endl << endl;
		}

		else if (layer_type[i] == DROPOUT) {
	//	checkCudaErrors(cudaEventRecord(start));//记录当前时间
	//	float milli = 0.0;
			DropoutLayerParams* cur_params = (DropoutLayerParams*)params[i];
			checkCUDNN(cudnnDropoutBackward(cudnn_handle, cur_params->dropout_desc,
				cur_params->input_tensor, dlayer_input[i + 1],
				cur_params->input_tensor, dlayer_input[i],
				cur_params->reserved_space, cur_params->reserved_space_size));
	//		checkCudaErrors(cudaEventRecord(stop)); //记录当前时间
	//		checkCudaErrors(cudaEventSynchronize(stop)); //等待事件结束 
	//		checkCudaErrors(cudaEventElapsedTime(&milli, start, stop)); //计时
	//		cout << i << "DROPOUT层反向传播的时间为(ms)：" <<milli << endl << endl;
		}

		else if (layer_type[i] == BATCHNORM) {
		checkCudaErrors(cudaEventRecord(start));//记录当前时间
		float milli = 0.0;
			BatchNormLayerParams* cur_params = (BatchNormLayerParams*)params[i];

			if (!pre_alloc_batch_norm_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				if (count <= 12) {
					cost = (st - space_tracker.free_bytes) / 1024 / 1024;
					if (cost > this->maxcost) {
						this->maxcost = cost;
					}
				}
				printf("某时刻占用的最大占用GPU内存为%f(MB)\n", this->maxcost);
				std::cout << "here\n";
				std::cout << "Free bytes: " << free_bytes << std::endl;
			}

			checkCUDNN(cudnnBatchNormalizationBackward(cudnn_handle, cur_params->mode,
				&alpha, &beta,
				&alpha, &beta,
				cur_params->input_tensor, layer_input[i],
				cur_params->input_tensor, dlayer_input[i + 1],
				cur_params->input_tensor, dlayer_input[i],
				cur_params->sbmv_desc, cur_params->scale,
				cur_params->dscale, cur_params->dbias,
				cur_params->epsilon,
				cur_params->result_save_mean, cur_params->result_save_inv_var));

			cur_params->stepParams(cublas_handle, learning_rate);
			checkCudaErrors(cudaEventRecord(stop)); //记录当前时间
			checkCudaErrors(cudaEventSynchronize(stop)); //等待事件结束 
			checkCudaErrors(cudaEventElapsedTime(&milli, start, stop)); //计时
			cout << i << "BATCHNORM层反向传播的时间为(ms)：" << milli << endl << endl;
		}

		else if (layer_type[i] == POOLING) {
	//	checkCudaErrors(cudaEventRecord(start));//记录当前时间
		float milli = 0.0;
			PoolingLayerParams* cur_params = (PoolingLayerParams*)params[i];
					checkCUDNN(cudnnPoolingBackward(cudnn_handle, cur_params->pool_desc, &alpha,
						cur_params->output_tensor, layer_input[i + 1],
						cur_params->output_tensor, dlayer_input[i + 1],
						cur_params->input_tensor, layer_input[i],
						&beta,
						cur_params->input_tensor, dlayer_input[i]));
	//				checkCudaErrors(cudaEventRecord(stop)); //记录当前时间
	//				checkCudaErrors(cudaEventSynchronize(stop)); //等待事件结束 
	//				checkCudaErrors(cudaEventElapsedTime(&milli, start, stop)); //计时
	//				cout << i << "POOLING层反向传播的时间为(ms)：" << milli << endl << endl;
		}

		else if (layer_type[i] == ACTV) {
		checkCudaErrors(cudaEventRecord(start));//记录当前时间
		float milli = 0.0;
			ActivationLayerParams* cur_params = (ActivationLayerParams*)params[i];
			checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
				cur_params->input_tensor, layer_input[i + 1],
				cur_params->input_tensor, dlayer_input[i + 1],
				cur_params->input_tensor, layer_input[i],
				&beta,
				cur_params->input_tensor, dlayer_input[i]));
			continue;
			checkCudaErrors(cudaEventRecord(stop)); //记录当前时间
			checkCudaErrors(cudaEventSynchronize(stop)); //等待事件结束 
			checkCudaErrors(cudaEventElapsedTime(&milli, start, stop)); //计时
			cout << i << "ACTV层反向传播的时间为(ms)：" << milli << endl << endl;
		}

		else if (layer_type[i] == SOFTMAX) {
	//	checkCudaErrors(cudaEventRecord(start));//记录当前时间
		float milli = 0.0;
			// std::cout << "compute here\n";
			SoftmaxLayerParams* cur_params = (SoftmaxLayerParams*)params[i];

						checkCUDNN(cudnnSoftmaxBackward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
							cur_params->input_tensor, layer_input[i + 1],
							cur_params->input_tensor, dlayer_input[i + 1],
							&beta,
							cur_params->input_tensor, dlayer_input[i]));
						 std::cout << "compute here\n";
		//				 checkCudaErrors(cudaEventRecord(stop)); //记录当前时间
		//				 checkCudaErrors(cudaEventSynchronize(stop)); //等待事件结束 
		//				 checkCudaErrors(cudaEventElapsedTime(&milli, start, stop)); //计时
		//				 cout << i << "SOFTMAX层反向传播的时间为(ms)：" << milli << endl << endl;
			continue;
		}

		// ---------------------- lDNN start ----------------------

		 checkCudaErrors(cudaDeviceSynchronize());
		double start_time, end_time;
		checkCudaErrors(cudaStreamSynchronize(stream_compute));

		if (train)
			start_time = clock();

		checkCudaErrors(cudaStreamSynchronize(stream_memory));
		if (train) {
			end_time = clock();
			float lag = (end_time - start_time);
			bwd_ldnn_lag.insert(bwd_ldnn_lag.begin(), lag);
		}

		if (layer_type[i] == CONV) {
			checkCNMEM(cnmemFree(cur_workspace, NULL));
			//checkCudaErrors(cudaFree(cur_workspace));
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			std::cout << "here\n";
			std::cout << "Free bytes: " << space_tracker.free_bytes << std::endl;
			if (!pre_alloc_conv_derivative) {
				ConvLayerParams* cur_params = (ConvLayerParams*)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
				std::cout << "here\n";
				std::cout << "Free bytes: " << space_tracker.free_bytes << std::endl;
			}
		}
		else if (layer_type[i] == FULLY_CONNECTED) {
			if (!pre_alloc_fc_derivative) {
				FCLayerParams* cur_params = (FCLayerParams*)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
				std::cout << "here\n";
				std::cout << "Free bytes: " << space_tracker.free_bytes << std::endl;
			}
		}
		else if (layer_type[i] == BATCHNORM) {
			if (train == true && !pre_alloc_batch_norm_derivative) {
				BatchNormLayerParams* cur_params = (BatchNormLayerParams*)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
			}
		}
	
		count++;
		printf("i+1:%d", i + 1);
		(cnmemFree(layer_input[i + 1], NULL));//checkCNMEM释放不需要的层的数据
		//cudaFree(layer_input[i + 1]);
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		(cnmemFree(dlayer_input[i + 1], NULL));//1pCNMEMcheck
		//cudaFree(dlayer_input[i + 1]);
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		if (i == 0) {
			checkCNMEM(cnmemFree(layer_input[i], NULL));
			//cudaFree(layer_input[i]);
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
		}
		
		
		// ---------------------- lDNN end ------------------------
	}
	if (space_tracker.getConsumed() != 0) {
		std::cout << "Panic!! Space not updated properly\n";
	}

	// exit(0);
}


int NeuralNet::findPrefetchLayer(int cur_layer) {
	for (int i = cur_layer; i >= 0; i--) {   //优化，将cur_layer - 1更改为cur_layer
		if (to_offload[i] && !prefetched[i]) {
			//prefetched[i] = true;  优化
			return i;
		}
	//	else if (layer_type[i] == CONV) { 
		//	return -1;
		//}
	}
	return -1;
}
