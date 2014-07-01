#include <stdio.h>
#include "izhikevich.h"

__global__ void izhikevichUpdateCUDA(
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

	float v = nDynState[DYN_STATE_V];
	float u = nDynState[DYN_STATE_U];
	// synaptic current + thalamic input
	float I = Isyn[nId] + 5.0f;

	if(v >= 30.0f){
		v = nDynParam[DYN_PARAM_C];
		u = u + nDynParam[DYN_PARAM_D];

		// neuron is firing
		firing[nId] = 1.0f;
	}else{
		// not firing
		firing[nId] = 0.0f;
	}

	// update state
	v += 0.5f * (0.04f * v * v + 5.0f * v + 140 - u + I);
	v += 0.5f * (0.04f * v * v + 5.0f * v + 140 - u + I);
	u += nDynParam[DYN_PARAM_A] * (nDynParam[DYN_PARAM_B] * v - u);

	// write result
	nDynState[DYN_STATE_V] = v;
	nDynState[DYN_STATE_U] = u;
}

#define NUM_WARPS 32
int izhikevichUpdateState(
	int numNeurons,
	float * dynState,
	float * firing,
	const float * dynParam,
	const float * Isyn
){
	// reset CUDA error
	cudaGetLastError();

	// update neurons
	dim3 threads(32 * NUM_WARPS);
	dim3 grid((int) ceil((double) numNeurons / (32 * NUM_WARPS)));
	izhikevichUpdateCUDA<<<grid, threads>>>(
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
