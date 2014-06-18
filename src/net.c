#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu.h"
#include "io.h"
#include "net.h"

int netNew(net_t * pNet){
	// neuron parameters
	pNet->dynParam = (const float *) calloc(
		pNet->numNeurons * DYN_PARAM_LEN,
		sizeof(float)
	);
	if(!pNet->dynParam){
		printf("Could not allocate memory for neuron parameters\n");
		return -1;
	}

	// neuron state
	pNet->dynState = (float *) calloc(
		pNet->numNeurons * DYN_STATE_LEN,
		sizeof(float)
	);
	if(!pNet->dynState){
		printf("Could not allocate memory for neuron state\n");
		return -1;
	}

	// neuron firing
	pNet->firing = (float *) calloc(pNet->numNeurons, sizeof(float));
	if(!pNet->firing){
		printf("Could not allocate memory for neuron firing\n");
		return -1;
	}

	// synaptic current
	pNet->Isyn = (float *) calloc(pNet->numNeurons, sizeof(float));
	if(!pNet->Isyn){
		printf("Could not allocate memory for synaptic current\n");
		return -1;
	}

	// synapse table
	pNet->syn = (float *) calloc(
		(pNet->synSuper + 1 + pNet->synSub) * pNet->numNeurons,
		sizeof(float)
	);
	if(!pNet->syn){
		printf("Could not allocate memory for synapse matrix\n");
		return -1;
	}

	return 0;
}

void netUpdateCurrent(net_t * pNet){
	/*
	** TODO
	** change to banded matrix multiplication
	*/
	gpuMultiplyMV(
		pNet->syn,
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

	const float * syn = NULL;
	size_t synRows = hNet->synSuper + 1 + hNet->synSub;
	size_t synCols = hNet->numNeurons;
	gpuCopyMemoryToGPU(
		(const void *) hNet->syn,
		(void **) &syn,
		synRows * synCols * sizeof(float)
	);

	net_t dNet = {
		.numNeurons = hNet->numNeurons,
		.t = hNet->t,
		.dynParam = dynParam,
		.dynState = dynState,
		.firing = firing,
		.Isyn = Isyn,
		.syn = syn
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
		(float *) pNet->dynParam,
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
		(float *) pNet->syn,
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
