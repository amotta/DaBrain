#include <stdio.h>
#include "goldman.h"

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
	// firing?
	DYN_STATE_FIRE,
	// transmembrane current
	DYN_STATE_I,
	// synaptic current
	DYN_STATE_I_SYN,
	// Na channel activation
	DYN_STATE_M,
	// Na channel inactivation
	DYN_STATE_H,
	// K channel activation
	DYN_STATE_N,
	DYN_STATE_LEN
};

int goldUpdate(neuron_t * neuron);

const neuron_t neuronGold = {
	.name = "Hodgkin-Huxley Goldman",
	.paramLen = DYN_PARAM_LEN,
	.stateLen = DYN_STATE_LEN,
	// indices
	.fireIdx = DYN_STATE_FIRE,
	.iSynIdx = DYN_STATE_I_SYN,
	// update
	.update = goldUpdate
};

int goldUpdate(neuron_t * neuron){
	printf("Yeah!\n");
	return 0;
}

// Faraday constant (C / mol)
#define C_F 96485.34f
// ideal gas constant (V C / K mol)
#define C_R 8.31446f
// temperature (K)
#define C_T 295.0f
// xi
#define C_xi (96485.34f / (8.31446f * 295.0f))
// internal Na concentration (mol / m^3)
#define C_cNaI 14.0f
// external Na concentration (mol / m^3)
#define C_cNaO 114.0f
// internal K concentration (mol / m^3)
#define C_cKI 120.0f
// external K concentration (mol / m^3)
#define C_cKO 2.5f
// leakage reversal potential (V)
#define C_eL -70e-3f
// membrane capacitance (F / m^2)
#define C_Cm 7e-12f
// membrane area (C / mol)
#define C_A 100e-12f
// time step
#define C_DT 1e-6f

__global__ void goldmanUpdateCUDA(
	int numNeurons,
	float * dynState,
	float * firing,
	const float * dynParam,
	const float * Isyn
){
	// neuron id
	int nId = blockDim.x * blockIdx.x + threadIdx.x;

	// let's not exaggerate
	if(nId >= numNeurons) return;

	// current state
	float v = dynState[DYN_STATE_V * numNeurons + nId];
	float m = dynState[DYN_STATE_M * numNeurons + nId];
	float h = dynState[DYN_STATE_H * numNeurons + nId];
	float n = dynState[DYN_STATE_N * numNeurons + nId];

	// parameters
	const float gL   = dynParam[DYN_PARAM_GL   * numNeurons + nId];
	const float pNa  = dynParam[DYN_PARAM_PNA  * numNeurons + nId];
	const float pK   = dynParam[DYN_PARAM_PK   * numNeurons + nId];
	const float type = dynParam[DYN_PARAM_TYPE * numNeurons + nId];

	// total current (A / m^2)
	float Itotal;

	// transmembrane current (A / m^2)
	float Istim = Isyn[nId];

	// add stimulation
	if(type < 0.5){
		// excitatory neuron
		Istim += 5.5e-12f;
	}else{
		// inhibitory neuron
		Istim += 10e-12f;
	}

	float aboveThresh = false;
	for(int i = 0; i < 1000; i++){
		float expVal = expf(v * C_xi);
		
		// leakage current
		float Il = gL * (v - C_eL);

		float Ina = 0;
		float Ik = 0;

		/*
		** TODO
		** This is a very crude way to prevent a division by zero.
		** Possible solutions:
		** - Check for zero voltage before expVal
		** - Try to use de l'Hôpital's rule
		*/
		if(expVal != 1.0f){
			// Na current
			const float goldNa = (C_cNaO - C_cNaI * expVal) / (1.0f - expVal);
			Ina = C_A * C_F * C_xi * m * m * h * pNa * v * goldNa;

			// K current
			const float goldK = (C_cKO - C_cKI * expVal) / (1.0f - expVal);
			Ik = C_A * C_F * C_xi * n * n * pK * v * goldK;
		}

		// calculate total current
		Itotal = Istim - Ina - Ik - Il;

		// membrane voltage
		const float dv = C_DT / C_Cm * Itotal;

		// Na activation
		const float dm = C_DT * (
			// aight
			(1 - m) * 60000 * (v + 0.033f)
			/ (1 - expf(-(v + 0.033f) / 0.003f))

			// yes!
			+ m * 70000 * (v + 0.042f)
			/ (1 - expf((v + 0.042f) / 0.02f))
		);

		// Na inactivation
		h += C_DT * (
			- (1 - h) * 50000 * (v + 0.065f)
			/ (1 - expf((v + 0.065f) / 0.006f))
			- h * 2250
			/ (1 + expf(-(v + 0.01f) / 0.01f))
		);

		// K activation
		n += C_DT * (
			// wumbaba
			(1 - n) * 16000 * (v + 0.01f)
			/ (1 - expf(-(v + 0.01f) / 0.01f))
			+ n * 40000 * (v + 0.035f)
			/ (1 - expf((v + 0.035f) / 0.01f))
		);

		// membrane voltage
		v += dv;

		// check for action potential
		if(v >= -35e-3f){
			aboveThresh = true;
		}
	}

	// write back dynamics state
	dynState[DYN_STATE_V * numNeurons + nId] = v;
	dynState[DYN_STATE_I * numNeurons + nId] = Itotal;
	dynState[DYN_STATE_M * numNeurons + nId] = m;
	dynState[DYN_STATE_H * numNeurons + nId] = h;
	dynState[DYN_STATE_N * numNeurons + nId] = n;

	// write firing
	if(aboveThresh){
		firing[nId] = 1.0f;
	}else{
		firing[nId] = 0.0f;
	}
}

/*
** Benchmarking on a GeForce GTX 580 showed that best performance
** is achieved with 32 threads per warp and 20 warps per block.
** See commit 569c50a3eab78bd089a25d7c04d79a1103279a7e
*/
#define NUM_WARPS 20

int goldmanUpdateState(
	int numNeurons,
	float * dynState,
	float * firing,
	const float * dynParam,
	const float * Isyn
){
	// reset CUDA error
	cudaGetLastError();

	// block size
	int blockSize = 32 * NUM_WARPS;

	// update neurons
	dim3 threads(blockSize);
	dim3 grid((int) ceil((double) numNeurons / blockSize));
	goldmanUpdateCUDA<<<grid, threads>>>(
		numNeurons,
		dynState,
		firing,
		dynParam,
		Isyn
	);

	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		printf("Could not update neuron states. Error:\n");
		printf("%s", cudaGetErrorString(error));
		return -1;
	}

	return 0;
}

