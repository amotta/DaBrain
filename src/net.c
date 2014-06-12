#include "net.h"

void netNew(net_t * pNet){
	// alloc neurons
	pNet->neurons = calloc(pNet->numNeurons, sizeof(neuron_t));

	// alloc synapse table
	pNet->synapses = calloc(pNet->numNeurons, sizeof(float *));
	for(size_t i = 0; i < pNet->numNeurons; i++){
		pNet->synapses[i] = calloc(pNet->numNeurons, sizeof(float));
	}
}
