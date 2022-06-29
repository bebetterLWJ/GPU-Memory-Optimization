#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <time.h>
#include "solver.h"
#include "neural_net.cuh" //改进  
using namespace std;

typedef unsigned char uchar;



int reverseInt(int n) {
	int bytes = 4;
	unsigned char ch[4];
	for (int i = 0; i < bytes; i++) {
		ch[i] = (n >> i * 8) & 255;
	}
	int p = 0;
	for (int i = 0; i < bytes; i++) {
		p += (int)ch[i] << (bytes - i - 1) * 8;
	}
	return p;
}

void readMNIST(vector<vector<uchar> >& train_images, vector<vector<uchar> >& test_images, vector<uchar>& train_labels, vector<uchar>& test_labels) {
	string filename_train_images = "D:/MNIST_data/train-images.idx3-ubyte";
	string filename_train_labels = "D:/MNIST_data/train-labels.idx1-ubyte";

	string filename_test_images = "D:/MNIST_data/t10k-images.idx3-ubyte";
	string filename_test_labels = "D:/MNIST_data/t10k-labels.idx1-ubyte";

	// read train/test images
	for (int i = 0; i < 2; i++) {
		string filename;
		if (i == 0)
			filename = filename_train_images;
		else
			filename = filename_test_images;

		ifstream f(filename.c_str(), ios::binary);
		if (!f.is_open())
			printf("Cannot read MNIST from %s\n", filename.c_str());

		// read metadata
		int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
		f.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		f.read((char*)&n_images, sizeof(n_images));
		n_images = reverseInt(n_images);
		f.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		f.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		for (int k = 0; k < n_images; k++) {
			vector<uchar> temp;
			temp.reserve(n_rows * n_cols);
			for (int j = 0; j < n_rows * n_cols; j++) {
				uchar t = 0;
				f.read((char*)&t, sizeof(t));
				temp.push_back(t);
			}
			if (i == 0)
				train_images.push_back(temp);
			else
				test_images.push_back(temp);
		}
		f.close();

	}

	// read train/test labels
	for (int i = 0; i < 2; i++) {
		string filename;
		if (i == 0)
			filename = filename_train_labels;
		else
			filename = filename_test_labels;

		ifstream f(filename.c_str(), ios::binary);
		if (!f.is_open())
			printf("Cannot read MNIST from %s\n", filename.c_str());

		// read metadata
		int magic_number = 0, n_labels = 0;
		f.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		f.read((char*)&n_labels, sizeof(n_labels));
		n_labels = reverseInt(n_labels);

		for (int k = 0; k < n_labels; k++) {
			uchar t = 0;
			f.read((char*)&t, sizeof(t));
			if (i == 0)
				train_labels.push_back(t);
			else
				test_labels.push_back(t);
		}

		f.close();

	}
}

void printTimes(vector<float>& time, string filename);
void printlDNNLag(vector<vector<float> >& fwd_ldnn_lag, vector<vector<float> >& bwd_ldnn_lag, string filename);
int best(NeuralNet net, std::vector<LayerSpecifier>& layer_specifier, int batch_size, long long dropout_seed, float* f_train_images, int* f_train_labels, int num_train);

int main(int argc, char* argv[]) {
	int batch_size =128;  //原代码为128，我的电脑可能跑不了128
	// int num_train = 100 * batch_size, num_val = batch_size;
	 int rows = 227, cols = 227, channels = 3;// rows,cols该小是否就不会触发报错呢
	 int input_size = rows * cols * channels;
	int input_channels = rows * cols * channels;
	int num_train = 500, num_test = 50;
	//void *X_train = malloc(num_train * input_channels * sizeof(float));
	// int *y_train = (int *)malloc(num_train * sizeof(int));
	//void *X_val = malloc(num_val * input_channels * sizeof(float));
	// int *y_val = (int *)malloc(num_val * sizeof(int));
	 //for (int i = 0; i < num_train; i++) {
	 //	for (int j = 0; j < input_channels; j++)
	//		((float *)X_train)[i * input_channels + j] = (rand() % 1000) * 1.0 / 1000;
	///	y_train[i] = 0;
	// }

	 //for (int i = 0; i < num_val; i++) {
	 //	for (int j = 0; j < input_channels; j++)
 	//	((float *)X_val)[i * input_channels + j] = (rand() % 1000) * 1.0 / 1000;
	// 	y_val[i] = rand() % 2;
	// }

	 
	// vector<vector<uchar> > train_images, test_images;
	//vector<uchar> train_labels, test_labels;
	// 
	
	float* f_train_images, * f_test_images;
	int* f_train_labels, * f_test_labels;
	
	f_train_images = (float*)malloc(num_train * input_size * sizeof(float));
	f_train_labels = (int*)malloc(num_train * sizeof(int));
	f_test_images = (float*)malloc(num_test * input_size * sizeof(float));
	f_test_labels = (int*)malloc(num_test * sizeof(int));
//readMNIST(train_images, test_images, train_labels, test_labels);
//for (int i = 0 ; i < )
//f_train_images = (float*)&train_images;
//f_train_labels = (int*)&train_labels;
//f_test_images = (float*)&test_images;
//f_test_labels = (int*)&test_labels;
	float* mean_image;
	mean_image = (float*)malloc(input_size * sizeof(float));
	for (int i = 0; i < num_train; i++) {
			for (int j = 0; j < input_channels; j++)
	   		((float *)f_train_images)[i * input_channels + j] = (rand() % 1000) * 1.0 / 1000;
	   f_train_labels[i] = 0;
	    }

		for (int i = 0; i < num_test; i++) {
			for (int j = 0; j < input_channels; j++)
	  	((float *)f_test_images)[i * input_channels + j] = (rand() % 1000) * 1.0 / 1000;
	   	f_test_labels[i] = rand() % 2;
	    }
	for (int i = 0; i < input_size; i++) {
		mean_image[i] = 0;
		for (int k = 0; k < num_train; k++) {
			mean_image[i] += f_train_images[k * input_size + i];
		}
		mean_image[i] /= num_train;
	}


	for (int i = 0; i < num_train; i++) {
		for (int j = 0; j < input_size; j++) {
			f_train_images[i * input_size + j] -= mean_image[j];
		}
	}

	for (int i = 0; i < num_test; i++) {
		for (int j = 0; j < input_size; j++) {
			f_test_images[i * input_size + j] -= mean_image[j];
		}

	}
	//for (int i = 0; i < num_train * input_size; i++) {
	//	printf("train_images%f\n", f_train_images[i]);
	//	printf("f_test_images%f\n", f_test_images[i]);
	//}
//	for (int i = 0; i < num_train ; i++) {
	//	printf("f_train_labels%f\n", f_train_labels[i]);
		
	//}
	// int input_channels = rows * cols * channels * 3, hidden_channels1 = 50, hidden_channels2 = 100, output_channels = 10;
	// vector<LayerSpecifier> layer_specifier;
	// ConvDescriptor layer0;
	// LayerSpecifier temp;
	// layer0.initializeValues(1, 3, 3, 3, rows, cols, 1, 1, 1, 1);
	// temp.initPointer(CONV);
	// *((ConvDescriptor *)temp.params) = layer0;
	// layer_specifier.push_back(temp);
	// ActivationDescriptor layer0_actv;
	// layer0_actv.initializeValues(RELU, 3, rows, cols);
	// temp.initPointer(ACTV);
	// *((ActivationDescriptor *)temp.params) = layer0_actv;
	// layer_specifier.push_back(temp);

	// BatchNormDescriptor layer0_bn;

	// for (int i = 0; i < 200; i++) {
	// 	layer0_bn.initializeValues(BATCHNORM_SPATIAL, 1e-5, 0.1, 3, rows, cols);
	// 	temp.initPointer(BATCHNORM);
	// 	*((BatchNormDescriptor *)temp.params) = layer0_bn;
	// 	layer_specifier.push_back(temp);

	// 	layer0.initializeValues(3, 3, 3, 3, rows, cols, 1, 1, 1, 1);
	// 	temp.initPointer(CONV);
	// 	*((ConvDescriptor *)temp.params) = layer0;
	// 	layer_specifier.push_back(temp);
	// 	layer0_actv.initializeValues(RELU, 3, rows, cols);
	// 	temp.initPointer(ACTV);
	// 	*((ActivationDescriptor *)temp.params) = layer0_actv;
	// 	layer_specifier.push_back(temp);
	// }

	// PoolingDescriptor layer0_pool;
	// layer0_pool.initializeValues(3, 2, 2, rows, cols, 0, 0, 2, 2, POOLING_MAX);
	// temp.initPointer(POOLING);
	// *((PoolingDescriptor *)temp.params) = layer0_pool;
	// layer_specifier.push_back(temp);

	// layer0_bn.initializeValues(BATCHNORM_SPATIAL, 1e-5, 0.1, 3, rows / 2, cols / 2);
	// temp.initPointer(BATCHNORM);
	// *((BatchNormDescriptor *)temp.params) = layer0_bn;
	// layer_specifier.push_back(temp);

	// // DropoutDescriptor layer0_dropout;
	// // layer0_dropout.initializeValues(0.2, 3, rows / 2, cols / 2);
	// // temp.initPointer(DROPOUT);
	// // *((DropoutDescriptor *)temp.params) = layer0_dropout;
	// // layer_specifier.push_back(temp);

	// layer0.initializeValues(3, 3, 3, 3, rows / 2, cols / 2, 1, 1, 1, 1);
	// temp.initPointer(CONV);
	// *((ConvDescriptor *)temp.params) = layer0;
	// layer_specifier.push_back(temp);
	// layer0_actv.initializeValues(RELU, 3, rows / 2, cols / 2);
	// temp.initPointer(ACTV);
	// *((ActivationDescriptor *)temp.params) = layer0_actv;
	// layer_specifier.push_back(temp);

	// layer0_bn.initializeValues(BATCHNORM_SPATIAL, 1e-5, 0.1, 3, rows / 2, cols / 2);
	// temp.initPointer(BATCHNORM);
	// *((BatchNormDescriptor *)temp.params) = layer0_bn;
	// layer_specifier.push_back(temp);

	// FCDescriptor layer1;
	// layer1.initializeValues(input_channels, hidden_channels1);
	// temp.initPointer(FULLY_CONNECTED);
	// *((FCDescriptor *)(temp.params)) = layer1;
	// layer_specifier.push_back(temp);

	// temp.initPointer(ACTV);
	// ActivationDescriptor layer1_actv;
	// layer1_actv.initializeValues(RELU, hidden_channels1, 1, 1);
	// *((ActivationDescriptor *)temp.params) = layer1_actv;
	// layer_specifier.push_back(temp);

	// layer0_bn.initializeValues(BATCHNORM_PER_ACTIVATION, 1e-5, 0.1, hidden_channels1, 1, 1);
	// temp.initPointer(BATCHNORM);
	// *((BatchNormDescriptor *)temp.params) = layer0_bn;
	// layer_specifier.push_back(temp);

	// temp.initPointer(FULLY_CONNECTED);
	// FCDescriptor layer2;
	// layer2.initializeValues(hidden_channels1, output_channels);
	// *((FCDescriptor *)temp.params) = layer2;
	// layer_specifier.push_back(temp);

	// // temp.initPointer(FULLY_CONNECTED);
	// // FCDescriptor layer3;
	// // layer3.initializeValues(hidden_channels2, output_channels);
	// // *((FCDescriptor *)temp.params) = layer3;
	// // layer_specifier.push_back(temp);

	// temp.initPointer(SOFTMAX);
	// SoftmaxDescriptor smax;
	// smax.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, output_channels, 1, 1);
	// *((SoftmaxDescriptor *)(temp.params)) = smax;
	// layer_specifier.push_back(temp);

	// AlexNet/*   网络初始化

	vector<LayerSpecifier> layer_specifier;
	{
		ConvDescriptor layer0;
		layer0.initializeValues(3, 96, 11, 11, 227, 227, 0, 0, 4, 4);
		LayerSpecifier temp; 
		temp.initPointer(CONV);
			* ((ConvDescriptor*)temp.params) = layer0;
		layer_specifier.push_back(temp);
}
	
	{
		PoolingDescriptor layer1;
		layer1.initializeValues(96, 3, 3, 55, 55, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor*)temp.params) = layer1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor layer2;
		layer2.initializeValues(96, 256, 5, 5, 27, 27, 2, 2, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor*)temp.params) = layer2;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor layer3;
		layer3.initializeValues(96, 256, 5, 5, 27, 27, 2, 2, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor*)temp.params) = layer3;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor layer4;
		layer4.initializeValues(256, 3, 3, 27, 27, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor*)temp.params) = layer4;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor layer5;
		layer5.initializeValues(256, 384, 3, 3, 13, 13, 1, 1, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor*)temp.params) = layer5;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor layer6;
		layer6.initializeValues(384, 384, 3, 3, 13, 13, 1, 1, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor*)temp.params) = layer6;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor layer7;
		layer7.initializeValues(384, 256, 3, 3, 13, 13, 1, 1, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor*)temp.params) = layer7;
		layer_specifier.push_back(temp);
	}
	
	{
		PoolingDescriptor layer9;
		layer9.initializeValues(256, 3, 3, 13, 13, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor*)temp.params) = layer9;
		layer_specifier.push_back(temp);
	}
	{
		FCDescriptor layer10;
		layer10.initializeValues(9216, 4096);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor*)temp.params) = layer10;
		layer_specifier.push_back(temp);
	}
	
	
	{
		SoftmaxDescriptor layer11;
		layer11.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 1000, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(SOFTMAX);
		*((SoftmaxDescriptor*)temp.params) = layer11;
		layer_specifier.push_back(temp);
	}/*
	

	// VGG specification
	// Look at user_iface.h for function declaration to initialize values
	vector<LayerSpecifier> layer_specifier;   //LayerSpecifier是一个类
	{
		ConvDescriptor part0_conv0;
		part0_conv0.initializeValues(3, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part0_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part0_conv1;
		part0_conv1.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part0_conv1;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool0;
		pool0.initializeValues(64, 2, 2, 224, 224, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part1_conv0;
		part1_conv0.initializeValues(64, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part1_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part1_conv1;
		part1_conv1.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part1_conv1;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool1;
		pool1.initializeValues(128, 2, 2, 112, 112, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part2_conv0;
		part2_conv0.initializeValues(128, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part2_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part2_conv1;
		part2_conv1.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part2_conv1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part2_conv2;
		part2_conv2.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part2_conv2;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool2;
		pool2.initializeValues(256, 2, 2, 56, 56, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool2;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part3_conv0;
		part3_conv0.initializeValues(256, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part3_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part3_conv1;
		part3_conv1.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part3_conv1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part3_conv2;
		part3_conv2.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part3_conv2;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool3;
		pool3.initializeValues(512, 2, 2, 28, 28, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool3;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part4_conv0;
		part4_conv0.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part4_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part4_conv1;
		part4_conv1.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part4_conv1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part4_conv2;
		part4_conv2.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part4_conv2;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool3;
		pool3.initializeValues(512, 2, 2, 14, 14, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool3;
		layer_specifier.push_back(temp);
	}

	{
		FCDescriptor part5_fc0;
		part5_fc0.initializeValues(7 * 7 * 512, 4096, RELU);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = part5_fc0;
		layer_specifier.push_back(temp);
	}
	{
		FCDescriptor part5_fc1;
		part5_fc1.initializeValues(4096, 4096, RELU);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = part5_fc1;
		layer_specifier.push_back(temp);
	}
	{
		FCDescriptor part5_fc2;
		part5_fc2.initializeValues(4096, 1000);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = part5_fc2;
		layer_specifier.push_back(temp);
	}
	{
		SoftmaxDescriptor s_max;
		s_max.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 1000, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(SOFTMAX);
		*((SoftmaxDescriptor *)temp.params) = s_max;
		layer_specifier.push_back(temp);
	}
	*/

	lDNNConvAlgo ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL; //优先性能，也可以选择优先内存优化
	lDNNType ldnn_type = lDNN_ALL;//这里改变vdnn模式
	string filename("ldnn_conv"); 
	if (argc == 3) {
		filename.assign("ldnn");
		// argv[1] - layers to offload, argv[2] - conv algo to use
		if (strcmp(argv[1], "dyn") == 0) {
			ldnn_type = lDNN_DYN;
			filename.append("_dyn");
		}
		else if (strcmp(argv[1], "conv") == 0) {
			ldnn_type = lDNN_CONV;
			filename.append("_conv");
		}
		else if (strcmp(argv[1], "all") == 0) {
			ldnn_type = lDNN_NONE;
			filename.append("_all");
		}
		else {
			printf("invalid argument.. using ldnn dynamic\n");
			filename.assign("ldnn_dyn");
		}
	if ((strcmp(argv[1], "conv") == 0 || strcmp(argv[1], "all") == 0)) {
			if (strcmp(argv[2], "p") == 0) {
				ldnn_conv_algo = lDNN_PERFORMANCE_OPTIMAL;
				filename.append("_p");
			}
			else if (strcmp(argv[2], "m") == 0) {
				ldnn_conv_algo = lDNN_MEMORY_OPTIMAL;
				filename.append("_m");
			}
			else {
				printf("invalid argument.. using ldnn dynamic\n");
				filename.assign("ldnn_dyn");
			}
		}
	}
//	printf("是否允许ldnn获得本机gpu信息，这将决定ldnn是否可以运行,同意请输入1，不同意则输入1以外的数字");
//	int permit;
//	cin >> permit ;
//	if (permit == 1){
	//获取本机gpu信息
	cudaDeviceProp myCUDA;
	if (cudaGetDeviceProperties(&myCUDA, 0) == cudaSuccess)
	{
		printf("Using device %d:\n", 0);
		printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
			myCUDA.name, (int)myCUDA.totalGlobalMem, (int)myCUDA.major,
			(int)myCUDA.minor, (int)myCUDA.clockRate);
	}
	int threadsPerBlock = myCUDA.maxThreadsPerBlock ;
	int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
	std::cout << "Maxium number per block = " << threadsPerBlock << std::endl;
	std::cout << "Blocks per Grid = " << blocksPerGrid << std::endl;
	//}
	double start_time, end_time;
	

//尖括号中包括四种信息，<<<块个数，线程个数，动态分配共享内存，流>>>，其中动态分配共享内存和流不是必填项。确定块个数和线程个数的一般步骤为：
	//1） 先根据GPU设备的硬件资源确定一个块内的线程个数；再根据数据大小和每个线程处理数据个数确定块个数。
	
	
	long long dropout_seed = 64;
	float softmax_eps = 1e-8;
	float init_std_dev = 0.1;
	//对网络进行初始化
	NeuralNet net(layer_specifier, DATA_FLOAT, TENSOR_NCHW, softmax_eps, init_std_dev, ldnn_type, ldnn_conv_algo);
	int jud = 1;
	int pre = 0;
	size_t exp_max_consume, max_consume;
	while (jud  == 1|| jud == 0 && batch_size - pre > 1) {
		break;//
		size_t free_bytes, total_bytes;
		net.Initial(layer_specifier,batch_size, SGD, dropout_seed);
		net.maxcost = 0;
     printf("batch_size:%d max:%d\n",batch_size,net.max);
	 jud = net.lDNNOptimize(exp_max_consume, max_consume);//vdnnoptimize并没有消耗内存
		 net.netfree(layer_specifier);
		batch_size = net.batch_size;
		pre = net.pre;
		if (batch_size - pre < 1) {
			break;
		}
		cout << "jud:" << jud << endl;
		
	}
	printf("pre:%d", pre);
	//batch_size = best(net, layer_specifier, pre, dropout_seed,f_train_images,f_train_labels,num_train);
	
	printf("the best batch_size:%d",batch_size);
	batch_size = pre;
    batch_size =1;
	net.Initial(layer_specifier, batch_size, SGD, dropout_seed);
	net.lDNNOptimize(exp_max_consume, max_consume);
	//net.free_bytes = max_consume;
	net.batch_size = 1;
	for (int i = 0; i < net.num_layers; i++) {
		std::cerr << "to_offload[i] " << net.to_offload[i] << std::endl;
	}
	size_t cnmem_stream_memory_size = net.free_bytes;
	cnmemDevice_t cnmem_device;
	cnmem_device.device = 0;
	cnmem_device.size = cnmem_stream_memory_size;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;
	// do not allow call to cudaMalloc
	checkCNMEM(cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW)); //应该是类似显存池的操作
	
	

	for (int i = 0; i < net.num_layers; i++) {
		// allocate pinned memory in host
		if (net.to_offload[i])
			(cudaMallocHost(&net.h_layer_input[i], net.layer_input_size[i] * net.data_type_size));
	}
	

	printf("batch_size:%dpre:%d", batch_size, pre);	

	//初始化
	vector<float> loss;
	vector<float> time;
	vector<vector<float> > fwd_ldnn_lag, bwd_ldnn_lag;

	start_time = clock();
	int num_epoch = 1;//原代码为1000，我的电脑可能跑不了128
	double learning_rate = 1e-15;
	double learning_rate_decay = 0.9;
	Solver solver(&net, (void*)f_train_images, f_train_labels, (void*)f_train_images, f_train_labels, num_epoch, SGD, learning_rate, learning_rate_decay, num_train, num_train);
	solver.getTrainTime(loss, time, 100, fwd_ldnn_lag, bwd_ldnn_lag);
	end_time = clock();
	std::cout <<"total cost" << end_time - start_time <<"s"<< std::endl;
	printTimes(time, filename);
	printf("batch_size:%d", batch_size);
	printlDNNLag(fwd_ldnn_lag, bwd_ldnn_lag, filename);
	
}
void leftTry(int* batch_size, int left);
void rightTry(int* batch_size, int right);
void printMess(size_t max_consume, size_t exp_max_consume, NeuralNet net);

int best(NeuralNet net, std::vector<LayerSpecifier>&layer_specifier,int batch_size, long long dropout_seed,float * f_train_images,int * f_train_labels,int num_train) {
	int left = 0;
	
	printf("%d", batch_size);
	int right = batch_size;
	
	net.Initial(layer_specifier, batch_size, SGD, dropout_seed);
	size_t exp_max_consume, max_consume;
	net.lDNNOptimize(exp_max_consume, max_consume);
	net.batch_size = batch_size;
	size_t cnmem_stream_memory_size = net.free_bytes;
	cnmemDevice_t cnmem_device;
	cnmem_device.device = 0;
	cnmem_device.size = cnmem_stream_memory_size;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;
	// do not allow call to cudaMalloc
	checkCNMEM(cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW)); //应该是类似显存池的操作
	printf("batch_size:%d\n", batch_size);
	printMess(max_consume, exp_max_consume, net);
	for (int i = 0; i < net.num_layers; i++) {
		// allocate pinned memory in host
		if (net.to_offload[i])
			(cudaMallocHost(&net.h_layer_input[i], net.layer_input_size[i] * net.data_type_size));
	}
	vector<float> loss;
	vector<float> time;
	vector<vector<float> > fwd_ldnn_lag, bwd_ldnn_lag;
	double start_time, end_time;
	start_time = clock();
	int num_epoch = 1;//原代码为1000，我的电脑可能跑不了128
	double learning_rate = 1e-15;
	double learning_rate_decay = 0.9;
	Solver solver(&net, (void*)f_train_images, f_train_labels, (void*)f_train_images, f_train_labels, num_epoch, SGD, learning_rate, learning_rate_decay, num_train, num_train);
	solver.getTrainTime(loss, time, 100, fwd_ldnn_lag, bwd_ldnn_lag);
	cnmemFinalize();
	net.netfree(layer_specifier);

	end_time = clock();
	double res = end_time - start_time;
	std::cout << "total cost" << res << "s" << std::endl;
	double preright = res;
	double min = res;
	int mloc = batch_size;
	double mid;
	leftTry(&batch_size, left);
	bool l = true;
	bool r = false;
	vector<int> arr ,vector<int>larr ,vector<int> rarr;
	
	arr.push_back(batch_size);
	while (left == 0 || right != left) {
		net.maxcost = 0;
		printf("batch_size:%d ", batch_size);
		printf("left:%d ", left);
		printf("right:%d\n", right);
		larr.push_back(left);
		rarr.push_back(right);
		net.Initial(layer_specifier, batch_size, SGD, dropout_seed);
		size_t exp_max_consume, max_consume;
		net.lDNNOptimize(exp_max_consume, max_consume);
		
		cnmemDevice_t cnmem_device;

		size_t cnmem_stream_memory_size = net.free_bytes;
		cnmem_device.device = 0;
		cnmem_device.size = cnmem_stream_memory_size;
		cnmem_device.numStreams = 0;
		cnmem_device.streams = NULL;
		cnmem_device.streamSizes = NULL;
		// do not allow call to cudaMalloc
		checkCNMEM(cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW));
		printMess(max_consume, exp_max_consume, net);
		for (int i = 0; i < net.num_layers; i++) {
			// allocate pinned memory in host
			if (net.to_offload[i])
				(cudaMallocHost(&net.h_layer_input[i], net.layer_input_size[i] * net.data_type_size));
		}
		start_time = clock();
		solver.getTrainTime(loss, time, 100, fwd_ldnn_lag, bwd_ldnn_lag);
		cnmemFinalize();
		net.netfree(layer_specifier);
		
		end_time = clock();
		res = end_time - start_time;
		std::cout << "total cost" << res << "s" << std::endl;
		arr.push_back(batch_size);
		if (res > min) {
			if (l) {
				if (batch_size + 1 <= mloc && batch_size + 1 > left) {
					left = batch_size + 1;
				}	
				if (batch_size < right) {
					rightTry(&batch_size,right);
				}
				else {
					break;
				}
				l = false;
				r = true;
				continue;
			}
			if (r) {
				if (batch_size - 1 >= mloc && batch_size - 1 < right) {
					right = batch_size - 1 ;
				}
				
				
				if (batch_size > left) {
					leftTry(&batch_size, left);
				}
				else {
					break;
				}
				
				l = true;
				r = false;
			}
			
		}
		else {
			min = res;
			mloc = batch_size;
			mid = batch_size ;
			
				if (batch_size > left) {
					leftTry(&batch_size, left);
				l = true;
				r = false;
				continue;
				}
				
				
				if (batch_size < right) {
					rightTry(&batch_size, right);
					l = false;
					r = true;
				}
			
		}
	}
	for (int i = 0; i < larr.size(); i++) {
		cout << "batch_size:" << arr[i+1] << endl;
		cout << "left:" << larr[i] << endl;
		cout << "right:" << rarr[i] << endl;
	}
	return mloc;
}
void leftTry(int * batch_size , int left) {
	if (*batch_size - left > 20) {
		*batch_size -= 20;
	}
	else if (*batch_size - left > 10) {
		*batch_size -= 10;
	}
	else {
		*batch_size-=1;
	}
}
void rightTry(int* batch_size, int right) {
	if (right - *batch_size > 20) {
		*batch_size += 20;
	}
	else if (right - *batch_size > 10) {
		*batch_size += 10;
	}
	else {
		*batch_size++;
	}
}
void printMess(size_t max_consume, size_t exp_max_consume, NeuralNet net) {
	for (int i = 0; i < net.num_layers; i++) {
		std::cerr << "to_offload[i] " << net.to_offload[i] << std::endl;
	}
	std::cout << "actual_max_consume: " << max_consume << std::endl;
	std::cout << "exp_max_consume: " << exp_max_consume << std::endl;
	std::cout << "diff_max_consume(MB): " << (max_consume - exp_max_consume) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "exp_free_bytes(MB): " << (net.free_bytes + 1024 * 1024 * 600 - exp_max_consume) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "exp_total_consume(MB): " << (net.init_free_bytes - (net.free_bytes + 600 * 1024 * 1024 - exp_max_consume)) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "actual_total_consume(MB): " << (net.init_free_bytes - (net.free_bytes + 600 * 1024 * 1024 - max_consume)) / (1.0 * 1024 * 1024) << std::endl;
}
void printTimes(vector<float>& time, string filename) {
	float mean_time = 0.0;
	float std_dev = 0.0;
	int N = time.size();
	for (int i = 0; i < N; i++) {
		mean_time += time[i];
	}
	mean_time /= N;
	//for (int i = 0; i < N; i++) {
	//	std_dev += pow(time[i] - mean_time, 2);
//	}
//	std_dev /= N;
	//pow(std_dev, 0.5);
	cout << "平均一次前向传播与反向传播的时间为:(ms) " << mean_time << endl;
	//cout << "Standard deviation: " << std_dev << endl;

	filename.append(".dat");
	fstream f;
	f.open(filename.c_str(), ios_base::out);

	for (int i = 0; i < N; i++) {
		f << time[i] << endl;
	}
	f << "mean_time: " << mean_time <<"ms"<< endl;
	f << "standard_deviation: " << std_dev << endl;
	f.close();

}

void printlDNNLag(vector<vector<float> >& fwd_ldnn_lag, vector<vector<float> >& bwd_ldnn_lag, string filename) {
	filename.append("_lag.dat");

	fstream f;
	f.open(filename.c_str(), ios_base::out);

	int N = fwd_ldnn_lag.size();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < fwd_ldnn_lag[i].size(); j++) {
			f << "fwd" << j << ": " << fwd_ldnn_lag[i][j] << endl;
		}
		for (int j = 0; j < bwd_ldnn_lag[i].size(); j++) {
			f << "bwd" << j << ": " << bwd_ldnn_lag[i][j] << endl;
		}
		f << endl;
	}
	f.close();
}