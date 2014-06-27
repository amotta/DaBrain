#ifndef NEURON_H
#define NEURON_H

enum DYN_PARAM {
	// Leakage conductance
	DYN_PARAM_GL,
	// Na permeability
	DYN_PARAM_PNA,
	// K permeability
	DYN_PARAM_PK,
	// neuron type
	DYN_PARAM_TYPE,
	DYN_PARAM_LEN
};

enum DYN_STATE {
	// membrane voltage
	DYN_STATE_V,
	// transmembrane current
	DYN_STATE_I,
	// Na channel activation
	DYN_STATE_M,
	// Na channel inactivation
	DYN_STATE_H,
	// K channel activation
	DYN_STATE_N,
	DYN_STATE_LEN
};

#endif
