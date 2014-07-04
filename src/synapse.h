#ifndef SYNAPSE_H
#define SYNAPSE_H

#include "neuron.h"

// define type
typedef struct SYNAPSE synapse_t;

// structure for all synapse models
struct SYNAPSE {
	// model name
	const char * name;
	// update
	int (* update)(synapse_t * synapse, neuron_t * neuron);

	// number of neurons
	int numNeurons;
	// number of synapses
	int numSynapses;
	// synapse matrix
	float * mat;
};

int synapseNew(synapse_t * synapse, int numNeurons, int numSynapses);
int synapseRead(synapse_t * synapse, const char * synapseFile);
int synapseToGPU(synapse_t * synapse);
int synapseUpdate(synapse_t * synapse, neuron_t * neurons);

#endif
