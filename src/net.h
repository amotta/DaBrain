#ifndef NET_H
#define NET_H

#include "neuron.h"

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
	// synapse matrix
	const float * syn;
	// number of superdiagonals
	const int synSuper;
	// number of subdiagonals
	const int synSub;
} net_t;

int netNew(net_t * pNet);
void netInit(net_t * pNet);
void netUpdate(net_t * pNet);
net_t netCopyToGPU(const net_t * hNet);
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
