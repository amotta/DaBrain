#ifndef NET_H
#define NET_H

#include "neuron.h"
#include "syn.h"

/*
** Entire network of neurons
*/
typedef struct {
	// number of neurons
	const int numNeurons;
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
	// synapses
	syn_t syn;
} net_t;

int netNew(net_t * pNet);
int netToGPU(net_t * gpuNet);
void netInit(net_t * pNet);
int netUpdate(net_t * pNet);
int netRead(net_t * pNet);
int netReadSize(
	int * pNumNeurons,
	int * pNumSyn
);
#endif
