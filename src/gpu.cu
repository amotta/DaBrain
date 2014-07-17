#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "gpu.h"

static cublasHandle_t handle;

int gpuInit(){
	// we prefer L1 cache
	cudaError_t error;
	error = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	if(error != cudaSuccess){
		printf("Could not set cache config\n");
		return -1;
	}

	// init cuBLAS library
	cublasStatus_t status = cublasCreate(&handle);
	if(status != CUBLAS_STATUS_SUCCESS){
		printf("Failed to create cuBLAS handle\n");
		return -1;
	}

	return 0;
}

int gpuCopyTo(
	const size_t size,
	const void * hPtr,
	void ** dPtr
){
	cudaError_t error;

	// allocate memory on GPU
	error = cudaMalloc(dPtr, size);

	if(error != cudaSuccess){
		printf("Failed to allocate device memory\n");
		return -1;
	}

	// copy data to GPU
	error = cudaMemcpy(
		(void *) *dPtr,
		hPtr,
		size,
		cudaMemcpyHostToDevice
	);

	if(error != cudaSuccess){
		printf("Failed to copy data to GPU. Error:\n");
		printf("%s\n", cudaGetErrorString(error));
		return -1;
	}

	return 0;
}

int gpuCopyFrom(
	const size_t size,
	const void * dPtr,
	void * hPtr
){
	cudaError_t error = cudaSuccess;

	// copy data from GPU
	error = cudaMemcpy(
		hPtr,
		dPtr,
		size,
		cudaMemcpyDeviceToHost
	);

	if(error != cudaSuccess){
		printf("Failed to copy data to host. Error:\n");
		printf("%s\n", cudaGetErrorString(error));
		return -1;
	}

	return 0;
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

int gpuMultiplySV(
	int vecRows,
	const float * alpha,
	float * vec
){
	cublasStatus_t status;
	status = cublasSscal(
		handle,
		// vector size
		vecRows,
		// scaling factor
		alpha,
		// vector
		vec,
		// stride between elements
		1
	);

	if(status != CUBLAS_STATUS_SUCCESS){
		printf("Error in vector scaling\n");
		return -1;
	}

	return 0;
}
