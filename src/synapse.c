#include <stdlib.h>
#include "gpu.h"
#include "io.h"
#include "synapse.h"

int synapseNew(synapse_t * this, int numNeurons, int numSynapses){
	this->numNeurons = numNeurons;
	this->numSynapses = numSynapses;

	// allocate synapse matrix
	this->mat = calloc(
		numNeurons * numSynapses,
		sizeof(float)
	);

	return 0;
}

int synapseRead(synapse_t * this, const char * synapseFile){
	int error = ioReadMat(
		synapseFile,
		(float *) this->mat,
		this->numSynapses,
		this->numNeurons
	);
	if(error) return -1;

	return 0;
}

int synapseToGPU(synapse_t * this){
	float * gpuMat = NULL;
	gpuCopyMemoryToGPU(
		(const void *) this->mat,
		(void **) &gpuMat,
		this->numNeurons * this->numSynapses * sizeof(float)
	);

	// write back
	this->mat = gpuMat;

	return 0;
}

int synapseUpdate(synapse_t * this, neuron_t * neurons){
	int error = this->update(this, neurons);
	if(error) return -1;

	return 0;
}
