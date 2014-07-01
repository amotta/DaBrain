#ifndef IZHIKEVICH_H
#define IZHIKEVICH_H

#ifdef __cplusplus
extern "C" {
#endif

enum DYN_PARAM {
	DYN_PARAM_A,
	DYN_PARAM_B,
	// voltage reset
	DYN_PARAM_C,
	// recovery reset
	DYN_PARAM_D,
	DYN_PARAM_LEN
};

enum DYN_STATE {
	// membrane voltage
	DYN_STATE_V,
	// recovery variable
	DYN_STATE_U,
	DYN_STATE_LEN
};

int neuronUpdateState(
	int numNeurons,
	float * dynState,
	float * firing,
	const float * dynParam,
	const float * Isyn
);

#ifdef __cplusplus
}
#endif

#endif
