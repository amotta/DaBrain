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

void gpuMultiplyMV(
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
		return;
	}
}

__global__ void updateState(
	float * dynState,
	const float * dynParam
){
	// neuron id
	int nId = blockDim.x * blockIdx.x + threadIdx.x;

	// pointer to corresponding column
	float * nDynState = &dynState[DYN_STATE_LEN * nId];
	const float * nDynParam = &dynParam[DYN_PARAM_LEN * nId];

	float v = nDynState[DYN_STATE_V];
	float u = nDynState[DYN_STATE_U];
	float I = nDynState[DYN_STATE_I_SYN];

	if(v >= 30.0f){
		v = nDynParam[DYN_PARAM_C];
		u = u + nDynParam[DYN_PARAM_D];

		// neuron is firing
		nDynState[DYN_STATE_FIRING] = 1.0f;
	}else{
		// not firing
		nDynState[DYN_STATE_FIRING] = 0.0f;
	}

	// update state
	v += 0.5f * (0.04f * v * v + 5.0f * v + 140 - u + I + 5.0f);
	v += 0.5f * (0.04f * v * v + 5.0f * v + 140 - u + I + 5.0f);
	u += nDynParam[DYN_PARAM_A] * (nDynParam[DYN_PARAM_B] * v - u);

	// write result
	dynState[DYN_STATE_V] = v;
	dynState[DYN_STATE_U] = u;
}

#define BLOCK_SIZE (32 * 32)
void gpuUpdateState(
	int numNeurons,
	float * dynState,
	const float * dynParam
){
	dim3 threads(BLOCK_SIZE);
	dim3 grid((int) ceil((double) numNeurons / BLOCK_SIZE));

	updateState<<<grid, threads>>>(dynState, dynParam);
}
