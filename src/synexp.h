#ifndef SYNEXP_H
#define SYNEXP_H

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

int synExpUpdateState(
	int numNeurons,
	float * synState,
	const float * firing
);

int synExpUpdateCurrent(
	int numNeurons,
	float * Isyn,
	const float * synState
);

#ifdef __cplusplus
}
#endif

#endif
