//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//
//__global__ void printElements(int* input) {
//	int block_size = blockDim.x * blockDim.y * blockDim.z;
//	int offset_x = (blockIdx.x * block_size);
//	int offset_y = (blockIdx.y * (block_size * gridDim.x));
//	int offset_z = (blockIdx.z * (block_size * gridDim.x * gridDim.y));
//	int tid = offset_x + offset_y + offset_z + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y) + threadIdx.x;
//
//	printf("input: %d | tid: %d\n", input[tid], tid);
//}
//
//int main() {
//	int array_size = 64;
//	int byte_size = array_size * sizeof(int);
//	int* h_data = (int*) malloc(byte_size);
//
//	time_t t;
//	srand((unsigned)time(&t));
//
//	for (int i = 0; i < array_size; i++)
//	{
//		h_data[i] = (int)(rand() & 0xff);
//	}
//
//	int* d_data;
//
//	cudaMalloc((void**)& d_data, byte_size);
//	cudaMemcpy(d_data, h_data, byte_size, cudaMemcpyHostToDevice);
//
//	dim3 block(2, 2, 2);
//	dim3 grid(2, 2, 2);
//
//	printElements << < grid, block >> > (d_data);
//	cudaDeviceSynchronize();
//
//	cudaFree(d_data);
//	free(h_data);
//
//	cudaDeviceReset();
//	return 0;
// }