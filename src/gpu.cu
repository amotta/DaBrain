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

__global__ void updateState(
	int numNeurons,
	float * dynState,
	float * firing,
	const float * dynParam,
	const float * Isyn
){
	// neuron id
	int nId = blockDim.x * blockIdx.x + threadIdx.x;

	// let's not exaggerate
	if(nId >= numNeurons) return;

	// pointer to corresponding column
	float * nDynState = &dynState[DYN_STATE_LEN * nId];
	const float * nDynParam = &dynParam[DYN_PARAM_LEN * nId];

	// current state
	float v = nDynState[DYN_STATE_V];
	float n = nDynState[DYN_STATE_N];
	float m = nDynState[DYN_STATE_M];
	float h = nDynState[DYN_STATE_H];

	/*
	** Source:
	** Taken from Gerstner's homepage and
	** Arhem's Berkley Madonna simulation.
	*/

	// K channels
	const float gK = nDynParam[DYN_PARAM_GK];
	const float vK = -12.0f;

	// Na channels
	const float gNa = nDynParam[DYN_PARAM_GNA];
	const float vNa = 115.0f;

	// leakage
	const float gL = nDynParam[DYN_PARAM_GL];
	const float vL = 10.6f;

	// synaptic current + thalamic input
	float I = Isyn[nId] + 7.0f;

	float aboveThresh = false;
	for(int i = 0; i < 1000; i++){
		// membrane voltage
		v += 0.001f * (
			I
			- gK * n * n * n * n * (v - vK)
			- gNa * m * m * m * h * (v - vNa)
			- gL * (v - vL) 
		);

		if(v >= 50.0f){
			aboveThresh = true;
		}

		// K activation
		n += 0.001f * (
			0.01f * (10 - v) / (expf((10 - v) / 10) - 1) * (1 - n)
			- 0.125f * expf(-v / 80) * n
		);

		// Na activation
		m += 0.001f * (
			0.1f * (25 - v) / (expf((25 - v) / 10) - 1) * (1 - m)
			- 4 * expf(-v / 18) * m
		);

		// Na inactivation
		h += 0.001f * (
			0.07f * expf(-v / 20) * (1 - h)
			- 1 / (expf((30 - v) / 10) + 1) * h
		);
	}

	// write back dynamics state
	nDynState[DYN_STATE_V] = v;
	nDynState[DYN_STATE_N] = n;
	nDynState[DYN_STATE_M] = m;
	nDynState[DYN_STATE_H] = h;

	// write firing
	if(aboveThresh){
		firing[nId] = 1.0f;
	}else{
		firing[nId] = 0.0f;
	}
}

#define BLOCK_SIZE (32 * 32)
int gpuUpdateState(
	int numNeurons,
	float * dynState,
	float * firing,
	const float * dynParam,
	const float * Isyn
){
	// reset CUDA error
	cudaGetLastError();

	// update neurons
	dim3 threads(BLOCK_SIZE);
	dim3 grid((int) ceil((double) numNeurons / BLOCK_SIZE));
	updateState<<<grid, threads>>>(
		numNeurons,
		dynState,
		firing,
		dynParam,
		Isyn
	);

	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		printf("Could not update neuron states. Error:\n");
		printf("%s", cudaGetErrorString(error));
		return -1;
	}

	return 0;
}
