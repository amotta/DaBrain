#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "net.h"

/*
** Build configuration for excitatory neuron.
*/
neuron_t neuronBuildExc(float r){
	neuronDynParam_t dynParam = {
		.a = 0.02,
		.b = 0.2,
		.c = -65 + 15 * powf(r, 2),
		.d = 8 - 6 * powf(r, 2)
	};

	neuronDynState_t dynState = {
		.v = -65,
		.u = -65 * dynParam.b,
		.firing = false
	};

	neuron_t neuron = {
		.dynParam = dynParam,
		.dynState = dynState
	};

	return neuron;
}

/*
** Build configuration for inhibitory neuron.
*/
neuron_t neuronBuildInh(float r){
	neuronDynParam_t dynParam = {
		.a = 0.02 + 0.08 * r,
		.b = 0.25 - 0.05 * r,
		.c = -65,
		.d = 2
	};

	neuronDynState_t dynState = {
		.v = -65,
		.u = -65 * dynParam.b,
		.firing = false
	};

	neuron_t neuron = {
		.dynParam = dynParam,
		.dynState = dynState
	};

	return neuron;
}

#define FRAC_EXC 0.8
void netBuild(net_t * pNet){
	size_t numExc = FRAC_EXC * pNet->numNeurons;
	
	// build excitatory neurons
	for(size_t i = 0; i < numExc; i++){
		pNet->neurons[i] = neuronBuildExc((float) rand() / RAND_MAX);
		for(size_t s = 0; s < pNet->numNeurons; s++){
			pNet->synapses[s][i] = +0.5 * rand() / RAND_MAX;
		}
	}

	// build inhibitory neurons
	for(size_t i = numExc; i < pNet->numNeurons; i++){
		pNet->neurons[i] = neuronBuildInh((float) rand() / RAND_MAX);
		for(size_t s = 0; s < pNet->numNeurons; s++){
			pNet->synapses[s][i] = -1.0 * rand() / RAND_MAX;
		}
	}
}

#define TIME_STEP 1.0
#define UPDATE_STEP 0.5
void neuronUpdate(neuron_t * pNeuron){
	// reset if neuron fired in previous step..
	if(pNeuron->dynState.firing){
		// .. reset voltage and recovery
		pNeuron->dynState.v = pNeuron->dynParam.c;
		pNeuron->dynState.u += pNeuron->dynParam.d;
		pNeuron->dynState.firing = false;
	}

	// update membrane voltage	
	for(size_t i = 0; i < 2; i++){
		pNeuron->dynState.v += UPDATE_STEP * (
			// membrane voltage
			0.04 * powf(pNeuron->dynState.v, 2)
			+ 5 * pNeuron->dynState.v
			+ 140
			// recovery
			- pNeuron->dynState.u
			// synaptic current
			+ pNeuron->dynState.I
		);
	}

	// update recovery
	pNeuron->dynState.u += pNeuron->dynParam.a * (
		pNeuron->dynParam.b * pNeuron->dynState.v
		- pNeuron->dynState.u
	);

	if(pNeuron->dynState.v >= 30){
		pNeuron->dynState.firing = true;
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

void usage(){
	printf("Number of neurons missing\n");
}

int main(int argc, char ** argv){
	if(argc < 2){
		usage();
		return EXIT_SUCCESS;
	}

	size_t numNeurons = strtol(argv[1], NULL, 10);
	net_t net = {
		.numNeurons = numNeurons
	};

	netNew(&net);

	netBuild(&net);
	printf("Net build\n");
	printf("Size: %lu\n", net.numNeurons);
	
	clock_t tic = clock();	
	for(size_t i = 0; i < 1000; i++){
		netUpdate(&net);
	}
	clock_t toc = clock();

	printf("Net updated\n");
	printf("Time elapsed: %f\n", (double) (toc - tic) / CLOCKS_PER_SEC);	
	return EXIT_SUCCESS;
}
