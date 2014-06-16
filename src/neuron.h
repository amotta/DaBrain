#ifndef NEURON_H
#define NEURON_H

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

#endif
