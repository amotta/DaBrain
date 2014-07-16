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

	// synapses
	int error = synNew(&pNet->syn);

	if(error){
		printf("Could not allocate memory for synapses\n");
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

	// write back
	gpuNet->dynParam = dynParam;
	gpuNet->dynState = dynState;
	gpuNet->firing = firing;
	gpuNet->Isyn = Isyn;

	return 0;
}

int netUpdateCurrent(net_t * pNet){
/*
	int error;
	error = gpuMultiplyBMV(
		// synapse matrix
		pNet->syn,
		pNet->numNeurons,
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
*/

	return 0;
}

int netUpdateState(net_t * pNet){
	int error;

	#ifdef MODEL_IZHIKEVICH
	error = izhikevichUpdateState(
		pNet->numNeurons,
		pNet->dynState,
		pNet->firing,
		pNet->dynParam,
		pNet->Isyn
	);
	#endif

	#ifdef MODEL_HODGKINHUXLEY
	error = hodgkinhuxleyUpdateState(
		pNet->numNeurons,
		pNet->dynState,
		pNet->firing,
		pNet->dynParam,
		pNet->Isyn
	);
	#endif

	#ifdef MODEL_GOLDMAN
	error = goldmanUpdateState(
		pNet->numNeurons,
		pNet->dynState,
		pNet->firing,
		pNet->dynParam,
		pNet->Isyn
	);
	#endif
	return error;
}

int netUpdate(net_t * pNet){
	int error;

	error = netUpdateCurrent(pNet);
	if(error) return error;

	error = netUpdateState(pNet);
	if(error) return error;

	return 0;
}

static const char * dynParamFile = "dynParam.bin";
static const char * dynStateFile = "dynState.bin";

int netRead(net_t * pNet){
	int error;

	// read dynamics parameters
	error = ioReadMat(
		dynParamFile,
		pNet->numNeurons,
		DYN_PARAM_LEN,
		(float *) pNet->dynParam
	);
	if(error) return error;

	// read dynamics state matrix
	error = ioReadMat(
		dynStateFile,
		pNet->numNeurons,
		DYN_STATE_LEN,
		pNet->dynState
	);
	if(error) return error;

	error = synRead(&pNet->syn);
	if(error) return -1;

	return 0;
}

int neuronReadSize(int * pNumNeurons){
	int error;
	int rows;
	int cols;

	// check neuron parameters
	error = ioReadMatSize(dynParamFile, &rows, &cols);
	if(error) return -1;

	if(cols != DYN_PARAM_LEN){
		printf("Invalid column count in %s\n", dynParamFile);
		return -1;
	}

	// this should be a constant
	const int numNeurons = rows;

	// check neuron state
	error = ioReadMatSize(dynStateFile, &rows, &cols);
	if(error) return -1;

	if(rows != numNeurons){
		printf("Invalid rows count in %s\n", dynStateFile);
		return -1;
	}

	if(cols != DYN_STATE_LEN){
		printf("Invalid column count in %s\n", dynStateFile);
		return -1;
	}

	// write back
	*pNumNeurons = numNeurons;

	// report success
	return 0;
}

int netReadSize(
	int * pNumNeurons,
	int * pNumSyn
){
	int error;

	// read neurons
	int neurons;
	error = neuronReadSize(&neurons);

	// check for error
	if(error) return -1;

	// should be a constant
	const int numNeurons = neurons;

	// read synapses
	int syn;
	error = synReadSize(&neurons, &syn);

	// check for error
	if(error) return -1;

	// should be a constant
	const int numSyn = syn;

	// neuron count must agree
	if(neurons != numNeurons){
		printf("Neuron count conflicting\n");
		return -1;
	}

	// write back
	*pNumNeurons = numNeurons;
	*pNumSyn = numSyn;

	return 0;
}
