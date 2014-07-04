#ifndef NEURON_H
#define NEURON_H

typedef struct NEURON neuron_t;

// structure for all neuron models
struct NEURON {
	// model name
	const char * name;
	// matrix size
	const int paramLen;
	const int stateLen;
	// data
	const float * param;
	float * state;
	// update function
	int (* updateFunc)(
		int numNeurons,
		neuron_t * neuron
	);
};

#endif
