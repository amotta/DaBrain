#ifndef SYN_H
#define SYN_H

#ifdef __cplusplus
extern "C" {
#endif

enum SYN_PARAM {
	SYN_PARAM_S,
	SYN_PARAM_F,
	SYN_PARAM_LEN
};

enum SYN_STATE {
	// slow
	SYN_STATE_S,
	// fast
	SYN_STATE_F,
	// activity
	SYN_STATE_A,
	SYN_STATE_LEN
};

enum SYN_TYPE {
	// excitatory
	SYN_TYPE_EXC,
	// inhibitory
	SYN_TYPE_INH,
	SYN_TYPE_LEN
};

typedef struct {
	// number of neurons
	const int numNeurons;
	// number of synapses
	const int numSyn;
	// synapse matrices
	const float ** synMats;
	// synapse parameters
	const float * synParam;
	// synapse state
	float * synState;
} syn_t;

int synNew(syn_t * syn);
int synRead(syn_t * syn);
int synReadSize(
	int * pNumNeurons,
	int * pNumSyn
);
int synUpdateState(
	const float * firing,
	syn_t * syn
);

#ifdef __cplusplus
}
#endif

#endif
