//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//
//inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
//	if (code != cudaSuccess) {
//		fprintf(stderr, "Error: %s in file %s line %d", cudaGetErrorString(code), file, line);
//		if (abort) exit(code);
//	}
//}
//
//__global__ void sum_array(int* a, int* b, int* c, int* r, int size) {
//	int block_size = blockDim.x * blockDim.y * blockDim.z;
//	int offset_x = (blockIdx.x * block_size);
//	int offset_y = (blockIdx.y * (block_size * gridDim.x));
//	int offset_z = (blockIdx.z * (block_size * gridDim.x * gridDim.y));
//	int tid = offset_x + offset_y + offset_z + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y) + threadIdx.x;
//
//	if (tid < size) {
//		r[tid] = a[tid] + b[tid] + c[tid];
//	}
//}
//
//void compare_arrays(int* a, int* b, int size) {
//	for (size_t i = 0; i < size; i++)
//	{
//		if (a[i] != b[i]) {
//			printf("Arrays are different \n");
//			return;
//		}
//	}
//
//	printf("Arrays are the same. \n");
//}
//
//void sum_array_cpu(int* a, int* b, int* c, int* r, int size) {
//	for (size_t i = 0; i < size; i++)
//	{
//		r[i] = a[i] + b[i] + c[i];
//	}
//}
//
//int main() {
//	int array_size = 2;
//	for (size_t i = 0; i < 22; i++)
//	{
//		array_size *= 2;
//	}
//
//	printf("Array size: %d\n", array_size);
//
//	int byte_size = array_size * sizeof(int);
//	int* h_a = (int*) malloc(byte_size);
//	int* h_b = (int*) malloc(byte_size);
//	int* h_c = (int*) malloc(byte_size);
//	int* gpu_results = (int*)malloc(byte_size);
//	int* host_results = (int*)malloc(byte_size);
//
//	time_t t;
//	srand((unsigned)time(&t));
//
//	for (int i = 0; i < array_size; i++)
//	{
//		h_a[i] = (int)(rand() & 0xff);
//		h_b[i] = (int)(rand() & 0xff);
//		h_c[i] = (int)(rand() & 0xff);
//	}
//
//	int *d_a, *d_b, *d_c, *d_r;
//
//	gpuErrchk(cudaMalloc((void**)& d_a, byte_size));
//	gpuErrchk(cudaMalloc((void**)& d_b, byte_size));
//	gpuErrchk(cudaMalloc((void**)& d_c, byte_size));
//	gpuErrchk(cudaMalloc((void**)& d_r, byte_size));
//
//	clock_t htod_start, htod_end;
//	htod_start = clock();
//	gpuErrchk(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_c, h_c, byte_size, cudaMemcpyHostToDevice));
//	htod_end = clock();
//
//	dim3 block(256);
//	printf("Block size: %d \n", block.x);
//	dim3 grid((array_size/block.x) + 1);
//
//	clock_t gpu_start, gpu_end;
//	gpu_start = clock();
//	sum_array << < grid, block >> > (d_a, d_b, d_c, d_r, array_size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpu_end = clock();
//
//	clock_t dtoh_start, dtoh_end;
//	dtoh_start = clock();
//	gpuErrchk(cudaMemcpy(gpu_results, d_r, byte_size, cudaMemcpyDeviceToHost));
//	dtoh_end = clock();
//
//	clock_t cpu_start, cpu_end;
//	cpu_start = clock();
//	sum_array_cpu(h_a, h_b, h_c, host_results, array_size);
//	cpu_end = clock();
//
//	compare_arrays(host_results, gpu_results, array_size);
//	printf("CPU time: %4.6f \n", (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
//	printf("Host to device time: %4.6f \n", (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));
//	printf("Device to host time: %4.6f \n", (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));
//	printf("Kernel time: %4.6f \n", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
//	printf("Total GPU time: %4.6f \n", (double)((double)((gpu_end - gpu_start) + (htod_end - htod_start) + (dtoh_end - dtoh_start)) / CLOCKS_PER_SEC));
//
//	
//	gpuErrchk(cudaFree(d_a));
//	free(h_a);
//
//	gpuErrchk(cudaFree(d_b));
//	free(h_b);
//
//	gpuErrchk(cudaFree(d_c));
//	free(h_c);
//
//	gpuErrchk(cudaFree(d_r));
//	free(gpu_results);
//	free(host_results);
//
//	gpuErrchk(cudaDeviceReset());
//	return 0;
//}
//
