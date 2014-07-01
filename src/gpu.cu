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
// internal Na concentration (mol / m^3)
#define C_cNaI 14.0f
// external Na concentration (mol / m^3)
#define C_cNaO 114.0f
// internal K concentration (mol / m^3)
#define C_cKI 120.0f
// external K concentration (mol / m^3)
#define C_cKO 2.5f
// leakage reversal potential (V)
#define C_eL -70e-3f
// membrane capacitance (F / m^2)
#define C_Cm 7e-12f
// membrane area (C / mol)
#define C_A 100e-12f

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

	// current state
	float v = dynState[DYN_STATE_V * numNeurons + nId];
	float m = dynState[DYN_STATE_M * numNeurons + nId];
	float h = dynState[DYN_STATE_H * numNeurons + nId];
	float n = dynState[DYN_STATE_N * numNeurons + nId];

	// parameters
	float gL   = dynParam[DYN_PARAM_GL   * numNeurons + nId];
	float pNa  = dynParam[DYN_PARAM_PNA  * numNeurons + nId];
	float pK   = dynParam[DYN_PARAM_PK   * numNeurons + nId];
	float type = dynParam[DYN_PARAM_TYPE * numNeurons + nId];

	// total current (A / m^2)
	float Itotal;

	// transmembrane current (A / m^2)
	float Istim = Isyn[nId];

	// add stimulation
	if(type < 0.5){
		// excitatory neuron
		Istim += 5.5e-12f;
	}else{
		// inhibitory neuron
		Istim += 10e-12f;
	}

	float dt = 1e-6f;
	float aboveThresh = false;
	for(int i = 0; i < 1000; i++){
		float expVal = expf(v * C_xi);
		
		// leakage current
		float Il = gL * (v - C_eL);

		float Ina = 0;
		float Ik = 0;

		/*
		** TODO
		** This is a very crude way to prevent a division by zero.
		** Possible solutions:
		** - Check for zero voltage before expVal
		** - Try to use de l'Hôpital's rule
		*/
		if(expVal != 1.0f){
			// Na current
			float goldNa = (C_cNaO - C_cNaI * expVal) / (1.0f - expVal);
			Ina = C_A * C_F * C_xi * m * m * h * pNa * v * goldNa;

			// K current
			float goldK = (C_cKO - C_cKI * expVal) / (1.0f - expVal);
			Ik = C_A * C_F * C_xi * n * n * pK * v * goldK;
		}

		// calculate total current
		Itotal = Istim - Ina - Ik - Il;

		// membrane voltage
		v += dt / C_Cm * Itotal;

		// Na activation
		double dm = dt * (
			// aight
			(1 - m) * 60000 * (v + 0.033f)
			/ (1 - expf(-(v + 0.033f) / 0.003f))

			// yes!
			+ m * 70000 * (v + 0.042f)
			/ (1 - expf((v + 0.042f) / 0.02f))
		);

		// Na inactivation
		double dh = dt * (
			- (1 - h) * 50000 * (v + 0.065f)
			/ (1 - expf((v + 0.065f) / 0.006f))
			- h * 2250
			/ (1 + expf(-(v + 0.01f) / 0.01f))
		);

		// K activation
		double dn = dt * (
			// wumbaba
			(1 - n) * 16000 * (v + 0.01f)
			/ (1 - expf(-(v + 0.01f) / 0.01f))
			+ n * 40000 * (v + 0.035f)
			/ (1 - expf((v + 0.035f) / 0.01f))
		);

		/*
		** We should try to avoid this. Excessive use of registers
		** limits the degree of parallelization on GPGPU.
		*/
		if(isnan(dm) || isnan(dh) || isnan(dn)){
			// nothing
		}else{
			m += dm;
			h += dh;
			n += dn;
		}

		// check for action potential
		if(v >= -35e-3f){
			aboveThresh = true;
		}
	}

	// write back dynamics state
	dynState[DYN_STATE_V * numNeurons + nId] = v;
	dynState[DYN_STATE_I * numNeurons + nId] = Itotal;
	dynState[DYN_STATE_M * numNeurons + nId] = m;
	dynState[DYN_STATE_H * numNeurons + nId] = h;
	dynState[DYN_STATE_N * numNeurons + nId] = n;

	// write firing
	if(aboveThresh){
		firing[nId] = 1.0f;
	}else{
		firing[nId] = 0.0f;
	}
}

/*
** TODO
** Perform benchmarks and find optimal number of warps per SM.
** - Look into optimization tool from Oxford University
*/
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
