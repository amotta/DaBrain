#include <math.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "gpu.h"
#include "neuron.h"

static bool ready = false;
static cublasHandle_t handle;

void gpuInit(){
	cublasStatus_t status = cublasCreate(&handle);
	if(status != CUBLAS_STATUS_SUCCESS){
		printf("Failed to create cuBLAS handle\n");
		return;
	}

	ready = true;
}

void gpuCopyMemoryToGPU(const void * hPtr, void ** dPtr, size_t size){
	cudaError_t error = cudaSuccess;

	error = cudaMalloc(dPtr, size);
	if(error != cudaSuccess){
		printf("Failed to allocate device memory\n");
		return;
	}

	error = cudaMemcpy((void *) *dPtr, hPtr, size, cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
		printf("Failed to copy data to GPU. Error:\n");
		printf("%s\n", cudaGetErrorString(error));
		return;
	}
}

void gpuCopyMemoryFromGPU(const void * dPtr, void * hPtr, size_t size){
	cudaError_t error = cudaSuccess;

	error = cudaMemcpy(hPtr, dPtr, size, cudaMemcpyDeviceToHost);
	if(error != cudaSuccess){
		printf("Failed to copy data to host. Error:\n");
		printf("%s\n", cudaGetErrorString(error));
		return;
	}
}

int gpuMultiplyBMV(
	const float * mat,
	int matRows,
	int matCols,
	int matSuper,
	int matSub,
	const float * vecIn,
	int vecInStride,
	float * vecOut,
	int vecOutStride
){
	if(!ready){
		gpuInit();
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;

	cublasStatus_t status;
	status = cublasSgbmv(
		handle,
		// no transformation
		CUBLAS_OP_N,
		// matrix dimensions
		matRows, matCols,
		// lower and upper diagonals
		matSub, matSuper,
		// alpha
		&alpha,
		// matrix
		mat,
		// leading dimension of matrix
		matSuper + 1 + matSub,
		// vector
		vecIn,
		// vector stride
		vecInStride,
		// beta
		&beta,
		// output
		vecOut,
		// output stride
		vecOutStride
	);

	if(status != CUBLAS_STATUS_SUCCESS){
		printf("Error in banded matrix vector multiplication.\n");
		return -1;
	}

	return 0;
}

int gpuMultiplyMV(
	const float * mat,
	int matRows,
	int matCols,
	const float * vecIn,
	int vecInStride,
	float * vecOut,
	int vecOutStride
){
	if(!ready){
		gpuInit();
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;

	cublasStatus_t status;
	status = cublasSgemv(
		handle,
		// no transformation
		CUBLAS_OP_N,
		// dimensions of S
		matRows, matCols,
		// only product (alpha = 1)
		&alpha,
		// synapse matrix
		mat,
		// leading dimension of synapse matrix
		matRows, 
		// vector
		vecIn,
		// stride between elements
		vecInStride,
		// no addition (beta = 0)
		&beta,
		// result
		vecOut,
		// stride between elements
		vecOutStride
	);

	if(status != CUBLAS_STATUS_SUCCESS){
		printf("Error in matrix vector multiplication\n");
		return -1;
	}

	return 0;
}

