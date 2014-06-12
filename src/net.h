#ifndef NET_H
#define NET_H

#include <stdlib.h>
#include "neuron.h"

/*
** Entire network of neurons
*/
typedef struct {
	// number of neurons
	const size_t numNeurons;
	// array of all neurons
	neuron_t * neurons;
	// matrix with synapic strength
	float ** synapses;
} net_t;

void netNew(net_t * pNet);
void netInit(net_t * pNet);
void netUpdate(net_t * pNet);

#endif
