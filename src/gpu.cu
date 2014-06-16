#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu.h"

void gpuCopyMemory(const void * hPtr, void ** dPtr, size_t size){
	cudaError_t error = cudaSuccess;

	error = cudaMalloc(dPtr, size);
	if(error != cudaSuccess){
		printf("Failed to allocate device memory\n");
		return;
	}

	error = cudaMemcpy((void *) *dPtr, hPtr, size, cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
		printf("Failed to copy data. Error:\n");
		printf("%s\n", cudaGetErrorString(error));
		return;
	}
}