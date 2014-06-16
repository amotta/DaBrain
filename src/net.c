#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu.h"
#include "net.h"

void netNew(net_t * pNet){
	// neuron parameters
	pNet->dynParam = (const float *) calloc(
		pNet->numNeurons * DYN_PARAM_LEN,
		sizeof(float)
	);

	if(!pNet->dynParam){
		printf("Could not allocate memory for neuron parameters\n");
		return;
	}

	// neuron state
	pNet->dynState = (float *) calloc(
		pNet->numNeurons * DYN_STATE_LEN,
		sizeof(float)
	);

	if(!pNet->dynState){
		printf("Could not allocate memory for neuron state\n");
		return;
	}

	// synapse table
	pNet->S = (float *) calloc(
		pNet->numNeurons * pNet->numNeurons,
		sizeof(float)
	);

	if(!pNet->S){
		printf("Could not allocate memory for synapse matrix\n");
		return;
	}
}

void netInitDynParam(net_t * pNet){
	for(int n = 0; n < pNet->numExc; n++){
		float r = (float) rand() / RAND_MAX;
		float * param = (float *) &pNet->dynParam[n * DYN_PARAM_LEN];
		param[DYN_PARAM_A] = 0.02f;
		param[DYN_PARAM_B] = 0.2f;
		param[DYN_PARAM_C] = -65.0f + 15.0f * powf(r, 2);
		param[DYN_PARAM_D] = 8.0f - 6.0f * powf(r, 2);
	}

	for(int n = pNet->numExc; n < pNet->numNeurons; n++){
		float r = (float) rand() / RAND_MAX;
		float * param = (float *) &pNet->dynParam[n * DYN_PARAM_LEN];

		param[DYN_PARAM_A] = 0.02f + 0.08f * r;
		param[DYN_PARAM_B] = 0.25f - 0.05f * r;
		param[DYN_PARAM_C] = -65.0f;
		param[DYN_PARAM_D] = 2.0f;	
	}
}

void netInitDynState(net_t * pNet){
	for(int n = 0; n < pNet->numNeurons; n++){
		float r = (float) rand() / RAND_MAX;
		float * param = (float *) &pNet->dynParam[n * DYN_PARAM_LEN];
		float * state = (float *) &pNet->dynState[n * DYN_STATE_LEN];

		// membrane voltage
		state[DYN_STATE_V] = -65.0f;
		// recovery
		state[DYN_STATE_U] = -65.0f * param[DYN_PARAM_B];
		// firing
		state[DYN_STATE_FIRING] = 0.0f;
		
		// thalamic input
		if(n < pNet->numExc){
			state[DYN_STATE_I_THAL] = 5.0f * r;
		}else{
			state[DYN_STATE_I_THAL] = 2.0f * r;
		}

		// synaptic current
		state[DYN_STATE_I_SYN] = 0.0f;
	}
}

void netInitSynapse(net_t * pNet){
	float * S = (float *) pNet->S;

	for(int pre = 0; pre < pNet->numNeurons; pre++){
		for(int post = 0; post < pNet->numNeurons; post++){
			float r = (float) rand() / RAND_MAX;
			
			if(pre < pNet->numExc){
				S[pre * pNet->numNeurons + post] = +0.5f * r;
			}else{
				S[pre * pNet->numNeurons + post] = -1.0f * r;
			}
		}
	}
}

void netInit(net_t * pNet){
	netInitDynParam(pNet);
	netInitDynState(pNet);
	netInitSynapse(pNet);
}

void netUpdateCurrent(net_t * pNet){
	gpuMultiplyMV(
		pNet->S,
		pNet->numNeurons,
		pNet->numNeurons,
		(const float *) &pNet->dynState[DYN_STATE_FIRING],
		DYN_PARAM_LEN,
		(float *) &pNet->dynState[DYN_STATE_I_SYN],
		DYN_PARAM_LEN
	);
}

void netUpdateState(net_t * pNet){
	gpuUpdateState(
		pNet->numNeurons,
		pNet->dynState,
		pNet->dynParam
	);
}

void netUpdate(net_t * pNet){
	netUpdateCurrent(pNet);
	netUpdateState(pNet);

	// increment time
	pNet->t++;
}

net_t netCopyToGPU(const net_t * hNet){
	const float * dynParam = NULL;
	gpuCopyMemory(
		(const void *) hNet->dynParam,
		(void **) &dynParam,
		hNet->numNeurons * DYN_PARAM_LEN * sizeof(float)
	);

	float * dynState = NULL;
	gpuCopyMemory(
		(const void *) hNet->dynState,
		(void **) &dynState,
		hNet->numNeurons * DYN_STATE_LEN * sizeof(float)
	);

	const float * S = NULL;
	gpuCopyMemory(
		(const void *) hNet->S,
		(void **) &S,
		hNet->numNeurons * hNet->numNeurons * sizeof(float)
	);

	net_t dNet = {
		.numNeurons = hNet->numNeurons,
		.numExc = hNet->numExc,
		.t = hNet->t,
		.dynParam = dynParam,
		.dynState = dynState,
		.S = S
	};

	return dNet;
}
