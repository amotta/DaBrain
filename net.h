#ifndef NET_H
#define NET_H

#include "neuron.h"

/*
** Entire network of neurons
*/
typedef struct {
	// number of neurons
	size_t numNeurons;
	// array of all neurons
	neuron_t * neurons;
	// matrix with synapic strength
	float ** synapses;
} net_t;

#endif
