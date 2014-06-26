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

// Faraday constant (C / mol)
#define C_F 96485.34f
// ideal gas constant (V C / K mol)
#define C_R 8.31446f
// temperature (K)
#define C_T 295.0f
// xi
#define C_xi (96485.34f / (8.31446f * 295.0f))
// internal Na concentration (mol)
#define C_cNaI 0.014f
// external Na concentration (mol)
#define C_cNaO 0.1145f
// internal K concentration (mol)
#define C_cKI 0.120f
// external K concentration (mol)
#define C_cKO 0.0025f
// leakage reversal potential (V)
#define C_eL -0.070f
// membrane capacitance (F / m^2)
#define C_Cm 0.070f

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
	float m = nDynState[DYN_STATE_M];
	float h = nDynState[DYN_STATE_H];
	float n = nDynState[DYN_STATE_N];

	// parameters
	float gL = nDynParam[DYN_PARAM_GL];
	float pNa = nDynParam[DYN_PARAM_PNA];
	float pK = nDynParam[DYN_PARAM_PK];

	// stimulation current
	// float Istim = Isyn[nId] + 65.0f;
	// stimulation current (A / m^2)
	float Istim = 0.060f;

	float aboveThresh = false;
	for(int i = 0; i < 1; i++){
		float expVal = expf(v * C_xi);

		// leakage current
		float Il = 0; // gL * (v - C_eL);

		// Na current
		float goldNa = (C_cNaO - C_cNaI * expVal) / (1 - expVal);
		float Ina = C_F * C_xi * m * m * h * pNa * v * goldNa;

		// K current
		float goldK = (C_cKO - C_cKI * expVal) / (1 - expVal);
		float Ik = C_F * C_xi * n * n * pK * v * goldK;

		float dv = 0.001f / C_Cm * (Istim - Ina - Ik - Il);

		if(nId == 1){
			printf("expVal = %f\n", expVal);
			printf("Il = %f\n", Il);
			printf("goldNa = %f\n", goldNa);
			printf("Ina = %f\n", Ina);
			printf("goldK = %f\n", goldK);
			printf("Ik = %f\n", Ik);
			printf("dv = %f\n", dv);
		}

		// membrane voltage
		v += dv;

		// Na activation
		m += 0.001f * (
			(1 - m) * 60000 * (v + 0.033f)
			/ (1 - expf(-(v + 0.033f) / 0.003f))
			+ m * 70000 * (v + 0.042f)
			/ (1 - expf((v + 0.042f) / 0.02f))
		);

		// Na inactivation
		h += 0.001f * (
			- (1 - h) * 50000 * (v + 0.065f)
			/ (1 - expf((v + 0.065f) / 0.006f))
			- h * 2250
			/ (1 + expf(-(v + 0.01f) / 0.01f))
		);

		// K activation
		n += 0.001f * (
			(1 - n) * 16000 * (v + 0.01f)
			/ (1 - expf(-(v + 0.01f) / 0.01f))
			+ n * 40000 * (v + 0.035f)
			/ (1 - expf((v + 0.035f) / 0.01f))
		);

		// check for action potential
		if(v >= 50.0f){
			aboveThresh = true;
		}
	}

	if(nId == 1){
		printf("%f %f %f %f\n", v, m, h, n);
	}

	// write back dynamics state
	nDynState[DYN_STATE_V] = v;
	nDynState[DYN_STATE_M] = m;
	nDynState[DYN_STATE_H] = h;
	nDynState[DYN_STATE_N] = n;

	// write firing
	if(aboveThresh){
		firing[nId] = 1.0f;
	}else{
		firing[nId] = 0.0f;
	}
}

#define BLOCK_SIZE (16 * 32)
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
