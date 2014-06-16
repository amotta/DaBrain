#ifndef NET_H
#define NET_H

#include "neuron.h"

/*
** Entire network of neurons
*/
typedef struct {
	// number of neurons
	const int numNeurons;
	// count of excitatory neurons
	const int numExc; 
	// time
	int t;
	// neuron dynamics parameter
	const float * dynParam;
	// neuron state
	float * dynState;
	// neuron firing
	float * firing;
	// synaptic current
	float * Isyn;
	// synapse matrix
	const float * S;
} net_t;

void netNew(net_t * pNet);
void netInit(net_t * pNet);
void netUpdate(net_t * pNet);
net_t netCopyToGPU(const net_t * hNet);

#endif
