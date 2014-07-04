#ifndef SYNAPSE_H
#define SYNPASE_H

#include "neuron.h"

// define type
typedef struct SYNAPSE synapse_t;

// structure for all synapse models
struct SYNAPSE {
	// model name
	const char * name;
	// synapse matrix
	const float * mat;
	// number of synapses
	const int number;
	// update function
	int (* updateFunc)(
		int numNeurons,
		neuron_t * neurons,
		synapse_t * synapse
	);
};
