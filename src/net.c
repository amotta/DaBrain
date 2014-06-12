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

#define FRAC_EXC 0.8
void netInit(net_t * pNet){
	size_t numExc = FRAC_EXC * pNet->numNeurons;
	
	// build excitatory neurons
	for(size_t i = 0; i < numExc; i++){
		pNet->neurons[i] = neuronInitExc((float) rand() / RAND_MAX);
		for(size_t s = 0; s < pNet->numNeurons; s++){
			pNet->synapses[s][i] = +0.5 * rand() / RAND_MAX;
		}
	}

	// build inhibitory neurons
	for(size_t i = numExc; i < pNet->numNeurons; i++){
		pNet->neurons[i] = neuronInitInh((float) rand() / RAND_MAX);
		for(size_t s = 0; s < pNet->numNeurons; s++){
			pNet->synapses[s][i] = -1.0 * rand() / RAND_MAX;
		}
	}
}

void netUpdateCurrent(net_t * pNet){
	size_t numExc = FRAC_EXC * pNet->numNeurons;

	for(size_t i = 0; i < pNet->numNeurons; i++){
		float I = 0;

		// collect synaptic currents
		for(size_t s = 0; s < pNet->numNeurons; s++){
			if(pNet->neurons[s].dynState.firing){
				I += pNet->synapses[i][s];
			}
		}

		// simulate thalamic input
		if(i < numExc){
			I += 5.0 * rand() / RAND_MAX;
		}else{
			I += 2.0 * rand() / RAND_MAX;
		}

		// update current
		pNet->neurons[i].dynState.I = I;
	}
}

void netUpdate(net_t * pNet){
	// update currents
	netUpdateCurrent(pNet);

	// update state
	for(size_t i = 0; i < pNet->numNeurons; i++){
		neuronUpdate(&pNet->neurons[i]);
	}
}
