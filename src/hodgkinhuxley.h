#ifndef HODGKINHUXLEY_H
#define HODGKINHUXLEY_H

#ifdef __cplusplus
extern "C" {
#endif

enum DYN_PARAM {
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
	// transmembrane current
	DYN_STATE_I,
	// K channel activation
	DYN_STATE_N,
	// Na channel activation
	DYN_STATE_M,
	// Na channel inactivation
	DYN_STATE_H,
	DYN_STATE_LEN
};

int hodgkinhuxleyUpdateState(
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
