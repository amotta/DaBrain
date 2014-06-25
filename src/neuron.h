#ifndef NEURON_H
#define NEURON_H

enum DYN_PARAM {
	// membrane capacitance
	DYN_PARAM_CM,
	// maximum conductance for
	// K channels
	DYN_PARAM_GK,
	// Na channels
	DYN_PARAM_GNA,
	// leakage
	DYN_PARAM_GL,
	DYN_PARAM_LEN
};

enum DYN_STATE {
	// membrane voltage
	DYN_STATE_V,
	// K channel activation
	DYN_STATE_N,
	// Na channel activation
	DYN_STATE_M,
	// Na channel inactivation
	DYN_STATE_H,
	DYN_STATE_LEN
};

#endif
