#ifndef NEURON_H
#define NEURON_H

typedef struct NEURON neuron_t;

// structure for all neuron models
struct NEURON {
	// model name
	const char * name;
	// number of parameters
	const int paramLen;
	// number of states
	const int stateLen;
	// index of firing
	const int fireIdx;
	// index of synaptic current
	const int iSynIdx;
	// update
	int (* update)(neuron_t * neuron);

	// number of neurons
	int numNeurons;
	// neuron parameters
	float * param;
	// neuron state
	float * state;
	// firing
	float * fire;
	// synaptic current
	float * iSyn;
};

int neuronNew(
	neuron_t * neuron,
	int numNeurons
);
int neuronRead(
	neuron_t * neuron,
	const char * dynParamFile,
	const char * dynStateFile
);
int neuronToGPU(neuron_t * neuron);
int neuronUpdate(neuron_t * neuron);

#endif
