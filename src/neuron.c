#include <math.h>
#include <stdlib.h>
#include "neuron.h"

/*
** Build configuration for excitatory neuron.
*/
neuron_t neuronInitExc(float r){
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
neuron_t neuronInitInh(float r){
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
