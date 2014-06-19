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

int netToGPU(net_t * gpuNet){
	// neuronal parameters
	const float * dynParam = NULL;
	gpuCopyMemoryToGPU(
		(const void *) gpuNet->dynParam,
		(void **) &dynParam,
		gpuNet->numNeurons * DYN_PARAM_LEN * sizeof(float)
	);
	if(!dynParam){
		printf("Could not copy neuron parameters to GPU.\n");
		return -1;
	}

	// neuronal state
	float * dynState = NULL;
	gpuCopyMemoryToGPU(
		(const void *) gpuNet->dynState,
		(void **) &dynState,
		gpuNet->numNeurons * DYN_STATE_LEN * sizeof(float)
	);
	if(!dynState){
		printf("Could not copy neuron states to GPU.\n");
		return -1;
	}

	float * firing = NULL;
	gpuCopyMemoryToGPU(
		(const void *) gpuNet->firing,
		(void **) &firing,
		gpuNet->numNeurons * sizeof(float)
	);
	if(!firing){
		printf("Could not copy firing vector to GPU.\n");
		return -1;
	}

	float * Isyn = NULL;
	gpuCopyMemoryToGPU(
		(const void *) gpuNet->Isyn,
		(void **) &Isyn,
		gpuNet->numNeurons * sizeof(float)
	);
	if(!Isyn){
		printf("Could not copy synaptic currents to GPU.\n");
		return -1;
	}

	const float * syn = NULL;
	size_t synRows = gpuNet->synSuper + 1 + gpuNet->synSub;
	size_t synCols = gpuNet->numNeurons;
	gpuCopyMemoryToGPU(
		(const void *) gpuNet->syn,
		(void **) &syn,
		synRows * synCols * sizeof(float)
	);
	if(!syn){
		printf("Could not copy synapse matrix to GPU.\n");
		return -1;
	}

	// write back
	gpuNet->dynParam = dynParam;
	gpuNet->dynState = dynState;
	gpuNet->firing = firing;
	gpuNet->Isyn = Isyn;
	gpuNet->syn = syn;

	return 0;
}

int netUpdateCurrent(net_t * pNet){
	int error;
	error = gpuMultiplyBMV(
		// synapse matrix
		pNet->syn,
		pNet->synSuper + 1 + pNet->synSub,
		pNet->numNeurons,
		pNet->synSuper,
		pNet->synSub,
		// firing vector
		pNet->firing, 1,
		// synaptic current vector
		pNet->Isyn, 1
	);

	if(error){
		printf("Could not update synaptic currents.\n");
		return -1;
	}

	return 0;
}

int netUpdateState(net_t * pNet){
	gpuUpdateState(
		pNet->numNeurons,
		pNet->dynState,
		pNet->firing,
		pNet->dynParam,
		pNet->Isyn
	);

	return 0;
}

int netUpdate(net_t * pNet){
	int error;

	error = netUpdateCurrent(pNet);
	if(error) return error;

	error = netUpdateState(pNet);
	if(error) return error;

	return 0;
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
		pNet->synSuper + 1 + pNet->synSub,
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
