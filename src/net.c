#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu.h"
#include "io.h"
#include "net.h"

void netNew(net_t * pNet){
	// neuron parameters
	pNet->dynParam = (const float *) calloc(
		pNet->numNeurons * DYN_PARAM_LEN, sizeof(float)
	);
	if(!pNet->dynParam){
		printf("Could not allocate memory for neuron parameters\n");
		return;
	}

	// neuron state
	pNet->dynState = (float *) calloc(
		pNet->numNeurons * DYN_STATE_LEN, sizeof(float)
	);
	if(!pNet->dynState){
		printf("Could not allocate memory for neuron state\n");
		return;
	}

	// neuron firing
	pNet->firing = (float *) calloc(pNet->numNeurons, sizeof(float));
	if(!pNet->firing){
		printf("Could not allocate memory for neuron firing\n");
		return;
	}

	// synaptic current
	pNet->Isyn = (float *) calloc(pNet->numNeurons, sizeof(float));
	if(!pNet->Isyn){
		printf("Could not allocate memory for synaptic current\n");
		return;
	}

	// synapse table
	pNet->S = (float *) calloc(
		pNet->numNeurons * pNet->numNeurons, sizeof(float)
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
		(const float *) pNet->firing, 1,
		pNet->Isyn, 1
	);
}

void netUpdateState(net_t * pNet){
	gpuUpdateState(
		pNet->numNeurons,
		pNet->dynState,
		pNet->firing,
		pNet->dynParam,
		pNet->Isyn
	);
}

void netUpdate(net_t * pNet){
	netUpdateCurrent(pNet);
	netUpdateState(pNet);
}

net_t netCopyToGPU(const net_t * hNet){
	const float * dynParam = NULL;
	gpuCopyMemoryToGPU(
		(const void *) hNet->dynParam,
		(void **) &dynParam,
		hNet->numNeurons * DYN_PARAM_LEN * sizeof(float)
	);

	float * dynState = NULL;
	gpuCopyMemoryToGPU(
		(const void *) hNet->dynState,
		(void **) &dynState,
		hNet->numNeurons * DYN_STATE_LEN * sizeof(float)
	);

	float * firing = NULL;
	gpuCopyMemoryToGPU(
		(const void *) hNet->firing,
		(void **) &firing,
		hNet->numNeurons * sizeof(float)
	);

	float * Isyn = NULL;
	gpuCopyMemoryToGPU(
		(const void *) hNet->Isyn,
		(void **) &Isyn,
		hNet->numNeurons * sizeof(float)
	);

	const float * S = NULL;
	gpuCopyMemoryToGPU(
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
		.firing = firing,
		.Isyn = Isyn,
		.S = S
	};

	return dNet;
}

int netRead(
	net_t * pNet,
	const char * dynParamFile,
	const char * dynStateFile,
	const char * synapseFile
){
	int error;

	// read dynamics parameters
	error = ioReadMat(
		dynParamFile,
		(float *) pNet->dynParams,
		DYN_PARAM_LEN,
		pNet->numNeurons
	);
	if(error) return error;

	// read dynamics state matrix
	error = ioReadMat(
		dynStateFile,
		pNet->dynState,
		DYN_STATE_LEN,
		pNet->numNeurons
	);
	if(error) return error;

	// read synapse matrix
	error = ioReadMat(
		synapseFile,
		(float *) pNet->S,
		pNet->synSuper + pNet->synSub + 1,
		pNet->numNeurons
	);
	if(error) return error;

	return 0;
}

int netReadSize(
	int * pNumNeurons,
	int * pSynSuper,
	int * pSynSub,
	const char * dynParamFile,
	const char * dynStateFile,
	const char * synapseFile
){
	int error;

	/*
	** check dynamics parameter matrix
	*/
	int dynParamRows, dynParamCols;
	error = ioReadMatSize(
		dynParamFile,
		&dynParamRows,
		&dynParamCols
	);

	if(error) return error;
	if(dynParamRows != DYN_PARAM_LEN){
		printf("Invalid row count in %s\n", dynParamFile);
		return -1;
	}

	// set number of neurons
	const int numNeurons = dynParamCols;

	/*
	** check dynamics state matrix
	*/
	int dynStateRows, dynStateCols;
	error = ioReadMatSize(
		dynStateFile,
		&dynStateRows,
		&dynStateCols
	);

	if(error) return error;
	if(dynStateRows != DYN_STATE_LEN){
		printf("Invalid row count in %s\n", dynStateFile);
		return -1;
	}

	if(dynStateCols != numNeurons){
		printf("Invalid column count in %s\n", dynStateFile);
		return -1;
	}

	/*
	** check synapse matrix
	**
	** NOTICE
	** The synapse matrix is assumed to be banded with equal number of
	** superdiagonals and subdiagonals. Together with the main diagonal
	** this makes an odd number of rows in the matrix.
	*/
	int synapseRows, synapseCols;
	error = ioReadMatSize(
		synapseFile,
		&synapseRows,
		&synapseCols
	);

	if(error) return error;

	if(synapseRows % 2 == 0){
		printf("Expected odd number of rows in %s\n", synapseFile);
		return -1;
	}

	if(synapseCols != dynParamCols){
		printf("Invalid column count in %s\n", synapseFile);
	}

	const int synSuper = (synapseRows - 1) / 2;
	const int synSub = (synapseRows - 1) / 2;

	// write back
	*pNumNeurons = numNeurons;
	*pSynSuper = synSuper;
	*pSynSub = synSub;

	return 0;
}
