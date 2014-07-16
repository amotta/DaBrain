#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "gpu.h"
#include "io.h"
#include "syn.h"

int synNew(syn_t * syn){
	const float ** synMats = (const float **) calloc(
		SYN_TYPE_LEN,
		sizeof(const float *)
	);

	if(!synMats) return -1;

	// allocate synapse matrices
	for(int t = 0; t < SYN_TYPE_LEN; ++t){
		synMats[t] = (const float *) calloc(
			syn->numNeurons * syn->numSyn,
			sizeof(const float)
		);

		if(!synMats[t]) return -1;
	}

	const float * synParam = (const float *) calloc(
		SYN_PARAM_LEN * syn->numNeurons,
		sizeof(const float)
	);

	if(!synParam) return -1;

	float * synState = (float *) calloc(
		SYN_STATE_LEN * syn->numNeurons,
		sizeof(float)
	);

	if(!synState) return -1;

	// write back
	syn->synMats = synMats;
	syn->synParam = synParam;
	syn->synState = synState;

	return 0;
}

static const char * synMatFileNames[SYN_TYPE_LEN] = {
	"synExc.bin",
	"synInh.bin"
};

static const char * synParamFileName = "synParam.bin";
static const char * synStateFileName = "synState.bin";

int synCheckSize(syn_t * syn){
	int error;
	int rows = 0;
	int cols = 0;

	for(int t = 0; t < SYN_TYPE_LEN; ++t){
		error = ioReadMatSize(synMatFileNames[t], &rows, &cols);
		if(error) return -1;
		if(rows != syn->numSyn) return -1;
		if(cols != syn->numNeurons) return -1;
	}

	error = ioReadMatSize("synParam.bin", &rows, &cols);
	if(error) return -1;
	if(rows != SYN_TYPE_LEN) return -1;
	if(cols != SYN_PARAM_LEN) return -1;

	error = ioReadMatSize("synState.bin", &rows, &cols);
	if(error) return -1;
	if(rows != syn->numNeurons) return -1;
	if(cols != SYN_STATE_LEN) return -1;

	return 0;
}

int synRead(syn_t * syn){
	int error;

	// check size of matrices
	error = synCheckSize(syn);

	if(error){
		printf("Invalid matrix size detected\n");
		return -1;
	};

	// load files
	for(int t = 0; t < SYN_TYPE_LEN; ++t){
		error = ioReadMat(
			synMatFileNames[t],
			syn->numSyn,
			syn->numNeurons,
			(float *) syn->synMats[t]
		);

		if(error){
			printf("Error while reading %s\n", synMatFileNames[t]);
			return -1;
		}
	}

	// read synapse parameters
	error = ioReadMat(
		synParamFileName,
		SYN_TYPE_LEN,
		SYN_PARAM_LEN,
		(float *) syn->synParam
	);

	if(error){
		printf("Error while reading %s\n", synParamFileName);
		return -1;
	}

	// read synapse states
	error = ioReadMat(
		synStateFileName,
		syn->numNeurons,
		SYN_STATE_LEN,
		syn->synState
	);

	if(error){
		printf("Error while reading %s\n", synStateFileName);
		return -1;
	}

	return 0;
}

__global__ void synUpdateVec(
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
int synUpdateState(
	const int numNeurons,
	float * synState,
	const float * firing
){
	dim3 threads(32 * NUM_WARPS);
	dim3 blocks((int) ceil((double) numNeurons / (32 * NUM_WARPS)));

	// launch kernel
	synUpdateVec<<<blocks, threads>>>(
		numNeurons,
		synState,
		firing
	);

	return 0;
}

int synUpdateCurrent(
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
		const float * synActive = &synState[
			SYN_STATE_LEN * numNeurons * t
			+ SYN_STATE_A * numNeurons
		];

		gpuMultiplyBMV(
			// synapse matrix
			synMats[t],
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
