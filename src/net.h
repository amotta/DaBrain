#ifndef NET_H
#define NET_H

#include "neuron.h"
#include "syn.h"

/*
** Entire network of neurons
*/
typedef struct {
	const int numNeurons;
	// neurons
	neuron_t neurons;
	// synapses
	syn_t syn;
	// firing vector
	float * firing;
	// conductance
	float * cond;
} net_t;

int netNew(net_t * pNet);
int netCopyToGPU(net_t * gpuNet);
int netUpdate(net_t * pNet);
int netRead(net_t * pNet);
int netReadSize(
	int * pNumNeurons,
	int * pNumSyn
);
#endif
