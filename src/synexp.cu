#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "synexp.h"

// host memory for decay constants
static const float tau[] = {
	0.5, 0.25,
	0.3, 0.15
};

// constant memory for decay constants
__constant__ float gpuTau[SYN_PARAM_LEN * SYN_TYPE_LEN];

int synExpInit(){
	cudaError_t error;
	error = cudaMemcpyToSymbol(
		(const void *) gpuTau,
		(const void *) tau,
		SYN_STATE_LEN * sizeof(float)
	);

	if(error != cudaSuccess){
		printf("Failed to initialize synapse module\n");
		return -1;
	}

	return 0;
}

__global__ void synExpUpdateVec(
	const int numNeurons,	
	float * __restrict__ vecState,
	const float * __restrict__ vecReset
){
	// synapse id
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	// avoid running out of bounds
	if(id >= numNeurons) return;

	// load reset value to register
	float reset = vecReset[id];

	// reserve memory for states
	float state[SYN_PARAM_LEN];

	#pragma unroll
	for(int t = 0; t < SYN_TYPE_LEN; ++t){
		#pragma unroll
		for(int s = 0; s < SYN_PARAM_LEN; ++s){
			// load into register
			state[s] = vecState[
				SYN_STATE_LEN * numNeurons * t
				+ s * numNeurons
				+ id
			];

			// update value
			state[s] = gpuTau[
				SYN_PARAM_LEN * t + s
			] * state[s] * (1.0f - reset) + reset;

			// store
			vecState[
				SYN_STATE_LEN * numNeurons * t
				+ s * numNeurons
				+ id
			] = state[s];
		}

		// activity
		vecState[
			SYN_STATE_LEN * numNeurons * t
			+ SYN_STATE_A * numNeurons
			+ id
		] = state[SYN_STATE_S] - state[SYN_STATE_F];
	}
}

#define NUM_WARPS 32
int synExpUpdateState(
	const int numNeurons,
	float * synState,
	const float * firing
){
	dim3 threads(32 * NUM_WARPS);
	dim3 blocks((int) ceil((double) numNeurons / (32 * NUM_WARPS)));

	// launch kernel
	synExpUpdateVec<<<blocks, threads>>>(
		numNeurons,
		synState,
		firing
	);

	return 0;
}

int synExpUpdateCurrent(
	const int numNeurons,
	const float * synState,
	const float ** synMats,
	const int synSuper,
	const int synSub,
	float * Isyn
){
	cudaError_t error;

	float * gpuCond;
	error = cudaMalloc((void **) &gpuCond, numNeurons * sizeof(float));

	if(error){
		printf("Could not allocate memory\n");
		return -1;
	}

	for(int t = 0; t < SYN_TYPE_LEN; ++t){
		float * synActive = &synState[
			SYN_STATE_LEN * numNeurons * t
			+ SYN_STATE_A * numNeurons
		];

		gpuMultiplyBMV(
			// synapse matrix
			synMat[t],
			numNeurons,
			numNeurons,
			synSuper,
			synSub,
			// activity vector
			synActive, 1,
			// conductance vector
			gpuCond, 1
		);

		/*
		** TODO
		** I = conductance x bias
		** bias = voltage - reversal potential
		** Isyn = sum of all I
		*/
	}
	
	cudaFree(gpuCond);

	return 0;
}
