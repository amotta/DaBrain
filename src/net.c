#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu.h"
#include "io.h"
#include "net.h"

int netNew(net_t * pNet){
	int error;

	error = neuronNew(&pNet->neurons);
	if(error) return -1;

	error = synNew(&pNet->syn);
	if(error) return -1;

	// neuron firing
	float * firing;
	firing = (float *) calloc(pNet->numNeurons, sizeof(float));
	if(!firing) return -1;

	// conductance
	float * cond;
	cond = (float *) calloc(
		SYN_TYPE_LEN * pNet->numNeurons,
		sizeof(float)
	);

	if(!cond) return -1;

	// write back
	pNet->firing = firing;
	pNet->cond = cond;

	return 0;
}

int netCopyToGPU(net_t * gpuNet){
	int error;

	// neurons
	error = neuronCopyToGPU(&gpuNet->neurons);
	if(error) return -1;

	// synapses
	error = synCopyToGPU(&gpuNet->syn);
	if(error) return -1;

	// firing vector
	float * firing;
	error = gpuCopyTo(
		gpuNet->numNeurons * sizeof(float),
		(const void *) gpuNet->firing,
		(void **) &firing
	);

	if(error) return -1;

	// conductance
	float * cond;
	error = gpuCopyTo(
		gpuNet->numNeurons * SYN_TYPE_LEN * sizeof(float),
		(const void *) gpuNet->cond,
		(void **) &cond
	);

	if(error) return -1;

	// write back
	gpuNet->firing = firing;
	gpuNet->cond = cond;

	return 0;
}

int netUpdate(net_t * pNet){
	int error;

	// update synapses
	error = synUpdate(
		pNet->firing,
		&pNet->syn,
		pNet->cond
	);

	if(error){
		printf("Failed to update synapse state\n");
		return -1;
	}

	// update neurons
	error = neuronUpdate(
		pNet->cond,
		&pNet->neurons,
		pNet->firing
	);

	if(error){
		printf("Failed to update neuron state\n");
		return -1;
	}

	return 0;
}

int netRead(net_t * pNet){
	int error;

	error = neuronRead(&pNet->neurons);
	if(error) return -1;

	error = synRead(&pNet->syn);
	if(error) return -1;

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
