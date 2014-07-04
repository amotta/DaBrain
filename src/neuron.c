#include <stdlib.h>
#include "gpu.h"
#include "io.h"
#include "neuron.h"

int neuronUpdatePtr(neuron_t * this);

int neuronNew(neuron_t * this, int numNeurons){
	this->numNeurons = numNeurons;

	// allocate parameters
	this->param = (const float *) calloc(
		(size_t) this->numNeurons * this->paramLen,
		sizeof(float)
	);

	// allocate state
	this->state = (float *) calloc(
		(size_t) this->numNeurons * this->paramLen,
		sizeof(float)
	);

	return 0;
}

int neuronRead(
	neuron_t * this,
	const char * dynParamFile,
	const char * dynStateFile
){
	int error;

	// read dynamics parameters
	error = ioReadMat(
		dynParamFile,
		(float *) this->param,
		this->numNeurons,
		this->paramLen
	);

	if(error){
		printf("Could not read neuron parameters\n");
		return -1;
	}

	// read dynamics state matrix
	error = ioReadMat(
		dynStateFile,
		this->state,
		this->numNeurons,
		this->stateLen
	);

	if(error){
		printf("Could not read neuron state\n");
		return -1;
	}

	neuronUpdatePtr(this);

	return 0;
}

int neuronToGPU(neuron_t * this){
	float * gpuParam;
	gpuCopyMemoryToGPU(
		(const void *) this->param,
		(void **) &gpuParam,
		(size_t) this->numNeurons * this->paramLen * sizeof(float)
	);

	float * gpuState = NULL;
	gpuCopyMemoryToGPU(
		(const void *) this->state,
		(void **) &gpuState,
		(size_t) this->numNeurons * this->stateLen * sizeof(float)
	);

	// write back
	this->param = gpuParam;
	this->state = gpuState;

	neuronUpdatePtr(this);

	return 0;
}

int neuronUpdate(neuron_t * this){
	this->update(this);
}

int neuronUpdatePtr(neuron_t * this){
	this->fire = &this->state[
		this->numNeurons * this->fireIdx
	];

	this->iSyn = &this->state[
		this->numNeurons * this->iSynIdx
	];
}
