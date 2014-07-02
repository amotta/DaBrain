#include <stdio.h>
#include "hodgkinhuxley.h"

__global__ void hodgkinhuxleyUpdateCUDA(
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
	float n = dynState[DYN_STATE_N * numNeurons + nId];
	float m = dynState[DYN_STATE_M * numNeurons + nId];
	float h = dynState[DYN_STATE_H * numNeurons + nId];
	// conductances
	const float gK  = dynParam[DYN_PARAM_GK  * numNeurons + nId];
	const float gNa = dynParam[DYN_PARAM_GNA * numNeurons + nId];
	const float gL  = dynParam[DYN_PARAM_GL  * numNeurons + nId];

	/*
	** reversal potentials
	** Source:
	** Taken from Gerstner's homepage and
	** Arhem's Berkley Madonna simulation.
	*/
	const float vK = -12.0f;
	const float vNa = 115.0f;
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
	dynState[DYN_STATE_V * numNeurons + nId] = v;
	dynState[DYN_STATE_I * numNeurons + nId] = I;
	dynState[DYN_STATE_N * numNeurons + nId] = n;
	dynState[DYN_STATE_M * numNeurons + nId] = m;
	dynState[DYN_STATE_H * numNeurons + nId] = h;

	// write firing
	if(aboveThresh){
		firing[nId] = 1.0f;
	}else{
		firing[nId] = 0.0f;
	}
}

#define BLOCK_SIZE (32 * 32)
int hodgkinhuxleyUpdateState(
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
	hodgkinhuxleyUpdateCUDA<<<grid, threads>>>(
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
