#ifndef NEURON_H
#define NEURON_H

#include <stdbool.h>

/*
** This structure contains all parameters for the
** model of neuron dynamics.
** Ideally, these values are constant during simulation.
*/
typedef struct {
	float a;
	float b;
	// membrane voltage reset
	float c;
	// recovery reset
	float d;
} neuronDynParam_t;

/*
** The following structure represents the state.
** These values change during simulation.
*/
typedef struct {
	// membrane voltage
	float v;
	// recovery variable
	float u;
	// firing?
	bool firing;
	// synaptic current
	float I;
} neuronDynState_t;

/*
** All data for a single neuron is collected in this structure.
*/
typedef struct {
	neuronDynParam_t dynParam;
	neuronDynState_t dynState;
} neuron_t;

neuron_t neuronInitExc(float r);
neuron_t neuronInitInh(float r);
void neuronUpdate(neuron_t * pNeuron);

#endif
