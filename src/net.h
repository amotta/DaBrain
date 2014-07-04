#ifndef NET_H
#define NET_H

#include "neuron.h"
#include "synapse.h"

/*
** Entire network of neurons
*/
typedef struct {
	// time
	int time;
	// number of neurons
	const int numNeurons;
	// neuron model
	neuron_t neuron;
	// synapse model
	synapse_t synapse;
#if 0
	// neuron dynamics parameter
	const float * dynParam;
	// neuron state
	float * dynState;
	// neuron firing
	float * firing;
	// synaptic current
	float * Isyn;
	// synapse matrix
	const float * syn;
	// number of superdiagonals
	const int synSuper;
	// number of subdiagonals
	const int synSub;
#endif
} net_t;

int netNew(net_t * pNet);
int netToGPU(net_t * gpuNet);
void netInit(net_t * pNet);
int netUpdate(net_t * pNet);
int netRead(
	net_t * pNet,
	const char * dynParamFile,
	const char * dynStateFile,
	const char * synFile
);
int netReadSize(
	int * pNumNeurons,
	int * pSynSuper,
	int * pSynSub,
	const char * dynParamFile,
	const char * dynStateFile,
	const char * synFile
);
#endif
