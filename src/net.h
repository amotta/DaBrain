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
	int time;
	// neuron dynamics parameter
	float * dynParam;
	// neuron state
	float * dynState;
	// synapse matrix
	float * S;
} net_t;

void netNew(net_t * pNet);
void netInit(net_t * pNet);
void netUpdate(net_t * pNet);

#endif
