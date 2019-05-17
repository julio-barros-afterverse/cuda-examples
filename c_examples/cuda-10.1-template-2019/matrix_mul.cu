#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "Error: %s in file %s line %d", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void matrix_mul_gpu(int* a, int* b, int* r, int dim) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < dim && y < dim) {
		r[y * dim + x] = 0;

		for (int i = 0; i < dim; i++)
		{
			r[y * dim + x] += a[y * dim + i] * b[i * dim + x];
		}
	}
}

void compare_arrays(int* a, int* b, int size) {
	for (size_t i = 0; i < size; i++)
	{
		if (a[i] != b[i]) {
			printf("Arrays are different \n");
			return;
		}
	}

	printf("Arrays are the same. \n");
}

void matrix_mul_cpu(int* a, int* b, int* r, int dim) {

	for (int x = 0; x < dim; x++)
	{
		for (int y = 0; y < dim; y++)
		{
			r[y * dim + x] = 0;
			for (int i = 0; i < dim; i++)
			{
				r[y * dim + x] += a[y * dim + i] * b[i * dim + x];
			}
		}
	}
}

int main() {
	int matrix_dim = 2;
	for (size_t i = 0; i < 8; i++)
	{
		matrix_dim *= 2;
	}

	int matrix_size = matrix_dim * matrix_dim;

	printf("Matrix size: %d x %d\n", matrix_dim, matrix_dim);

	int byte_size = matrix_size * sizeof(int);
	int* h_a = (int*)malloc(byte_size);
	int* h_b = (int*)malloc(byte_size);
	int* gpu_results = (int*)malloc(byte_size);
	int* host_results = (int*)malloc(byte_size);

	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < matrix_size; i++)
	{
		h_a[i] = (int)(rand() & 0xff);
		h_b[i] = (int)(rand() & 0xff);
	}

	int* d_a, * d_b, * d_r;

	gpuErrchk(cudaMalloc((void**)& d_a, byte_size));
	gpuErrchk(cudaMalloc((void**)& d_b, byte_size));
	gpuErrchk(cudaMalloc((void**)& d_r, byte_size));

	clock_t htod_start, htod_end;
	htod_start = clock();
	gpuErrchk(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
	htod_end = clock();

	dim3 block(32, 32);
	printf("Block size: %d \n", block.x);
	dim3 grid(matrix_dim/block.x, matrix_dim/block.y);

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	matrix_mul_gpu <<< grid, block >>> (d_a, d_b, d_r, matrix_dim);
	gpuErrchk(cudaDeviceSynchronize());
	gpu_end = clock();

	clock_t dtoh_start, dtoh_end;
	dtoh_start = clock();
	gpuErrchk(cudaMemcpy(gpu_results, d_r, byte_size, cudaMemcpyDeviceToHost));
	dtoh_end = clock();

	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	matrix_mul_cpu(h_a, h_b, host_results, matrix_dim);
	cpu_end = clock();

	compare_arrays(host_results, gpu_results, matrix_size);
	printf("CPU time: %4.6f \n", (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
	printf("Host to device time: %4.6f \n", (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));
	printf("Device to host time: %4.6f \n", (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));
	printf("Kernel time: %4.6f \n", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
	printf("Total GPU time: %4.6f \n", (double)((double)((gpu_end - gpu_start) + (htod_end - htod_start) + (dtoh_end - dtoh_start)) / CLOCKS_PER_SEC));


	gpuErrchk(cudaFree(d_a));
	free(h_a);

	gpuErrchk(cudaFree(d_b));
	free(h_b);

	gpuErrchk(cudaFree(d_r));
	free(gpu_results);
	free(host_results);

	gpuErrchk(cudaDeviceReset());
	return 0;
}

