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
	// neuron model
	neuron_t neuron;
	// synapse model
	synapse_t synapse;
} net_t;

int netNew(
	net_t * this,
	int numNeurons,
	int numSynapses
);
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
