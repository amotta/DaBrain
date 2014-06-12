#ifndef NEURON_H
#define NEURON_H

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
} neuronDynState_t;

typedef struct {
	neuronDynParam_t dynParam;
	neuronDynState_t dynState;
} neuron_t;

#endif
