#include <math.h>
#include <stdio.h>
#include "gpu.h"
#include "io.h"
#include "neuron.h"
#include "syn.h"

static const char * dynParamFile = "dynParam.bin";
static const char * dynStateFile = "dynState.bin";

int neuronNew(neuron_t * neuron){
	// parameters
	const float * dynParam;
	dynParam = (const float *) calloc(
		neuron->numNeurons * DYN_PARAM_LEN,
		sizeof(float)
	);

	if(!dynParam) return -1;

	// state
	float * dynState;
	dynState = (float *) calloc(
		neuron->numNeurons * DYN_STATE_LEN,
		sizeof(float)
	);

	if(!dynState) return -1;

	// write back
	neuron->dynParam = dynParam;
	neuron->dynState = dynState;

	return 0;
}

int neuronCopyToGPU(neuron_t * neuron){
	int error;

 	// neuronal parameters
	const float * dynParam;
	error = gpuCopyTo(
		neuron->numNeurons * DYN_PARAM_LEN * sizeof(float),
		(const void *) neuron->dynParam,
		(void **) &dynParam
	);

	if(error){
		printf("Could not copy neuron parameters to GPU\n");
		return -1;
	}

	// neuronal state
	float * dynState;
	error = gpuCopyTo(
		neuron->numNeurons * DYN_STATE_LEN * sizeof(float),
		(const void *) neuron->dynState,
		(void **) &dynState
	);

	if(error){
		printf("Could not copy neuron states to GPU\n");
		return -1;
	}

	// write back
	neuron->dynParam = dynParam;
	neuron->dynState = dynState;

	return 0;
}

int neuronRead(neuron_t * neuron){
	int error;

	// read dynamics parameters
	error = ioReadMat(
		dynParamFile,
		neuron->numNeurons,
		DYN_PARAM_LEN,
		(float *) neuron->dynParam
	);

	if(error){
		printf("Failed to read %s\n", dynParamFile);
		return -1;
	}

	// read dynamics state matrix
	error = ioReadMat(
		dynStateFile,
		neuron->numNeurons,
		DYN_STATE_LEN,
		neuron->dynState
	);

	if(error){
		printf("Failed to read %s\n", dynStateFile);
		return -1;
	}

	return 0;
}

int neuronReadSize(int * pNumNeurons){
	int error;
	int rows;
	int cols;

	// check neuron parameters
	error = ioReadMatSize(dynParamFile, &rows, &cols);
	if(error) return -1;

	if(cols != DYN_PARAM_LEN){
		printf("Invalid column count in %s\n", dynParamFile);
		return -1;
	}

	// this should be a constant
	const int numNeurons = rows;

	// check neuron state
	error = ioReadMatSize(dynStateFile, &rows, &cols);
	if(error) return -1;

	if(rows != numNeurons){
		printf("Invalid rows count in %s\n", dynStateFile);
		return -1;
	}

	if(cols != DYN_STATE_LEN){
		printf("Invalid column count in %s\n", dynStateFile);
		return -1;
	}

	// write back
	*pNumNeurons = numNeurons;

	// report success
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
// excitatory reversal potential (V)
#define C_eExc 0.0f
// inhibitory reversal potential (V) 
#define C_eInh 0.0f
// membrane capacitance (F / m^2)
#define C_Cm 7e-12f
// membrane area (C / mol)
#define C_A 100e-12f

__global__ void neuronUpdateKernel(
	const int numNeurons,
	const float * __restrict__ cond,
	const float * __restrict__ dynParam,
	float * __restrict__ dynState,
	float * __restrict__ firingVec
){
	// neuron id
	const int nId = blockDim.x * blockIdx.x + threadIdx.x;

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

	// conductances
	const float gExc = cond[SYN_TYPE_EXC * numNeurons + nId];
	const float gInh = cond[SYN_TYPE_INH * numNeurons + nId];

	// total current (A / m^2)
	float Itotal;

	// stimulation current (A / m^2)
	float Istim = 0.0f;

	// add stimulation
	if(type < 0.5f){
		// excitatory neuron
		Istim = 5.5e-12f;
	}else{
		// inhibitory neuron
		Istim = 10e-12f;
	}

	float dt = 1e-6f;
	float firing = 0.0f;
	for(int i = 0; i < 1000; i++){
		float expVal = expf(v * C_xi);
		
		/*
		** TODO
		** This is a very crude way to prevent a division by zero.
		** Possible solutions:
		** - Check for zero voltage before expVal
		** - Try to use de l'Hôpital's rule
		*/
		float Ina;
		Ina  = C_A * C_F * C_xi * m * m * h * pNa;
		Ina *= C_cNaO - C_cNaI * expVal;

		float Ik;
		Ik  = C_A * C_F * C_xi * n * n * pK;
		Ik *= C_cKO - C_cKI * expVal;

		/*
		** Avoid division by zero and use de l'Hôpital's rule
		** to calculate Ina and Ik.
		*/
		if(expVal == 1.0f){
			Ina *= 1.0f / (1.0f - C_xi);
			Ik  *= 1.0f / (1.0f - C_xi);
		}else{
			Ina *= v / (1.0f - expVal);
			Ik  *= v / (1.0f - expVal);
		}

		// add stimulation current
		Itotal  = Istim;

		// add leakage, Na, and K current
		Itotal -= gL * (v - C_eL);

		// Na current
		Itotal -= Ina;

		// K+ current
		Itotal -= Ik;

		// add synaptic currents
		Itotal -= gExc * (v - C_eExc);
		Itotal -= gInh * (v - C_eInh);

		// membrane voltage
		float dv = dt / C_Cm * Itotal;

		// Na activation
		float dm = dt * (
			// aight
			(1 - m) * 60000 * (v + 0.033f)
			/ (1 - expf(-(v + 0.033f) / 0.003f))

			// yes!
			+ m * 70000 * (v + 0.042f)
			/ (1 - expf((v + 0.042f) / 0.02f))
		);

		// Na inactivation
		float dh = dt * (
			- (1 - h) * 50000 * (v + 0.065f)
			/ (1 - expf((v + 0.065f) / 0.006f))
			- h * 2250
			/ (1 + expf(-(v + 0.01f) / 0.01f))
		);

		// K activation
		float dn = dt * (
			// wumbaba
			(1 - n) * 16000 * (v + 0.01f)
			/ (1 - expf(-(v + 0.01f) / 0.01f))
			+ n * 40000 * (v + 0.035f)
			/ (1 - expf((v + 0.035f) / 0.01f))
		);

		// always update membrane voltage
		v += dv;

		// we should try to avoid this
		if(isnan(dm) || isnan(dh) || isnan(dn)){
			// nothing
		}else{
			m += dm;
			h += dh;
			n += dn;
		}

		// check for action potential
		if(v >= -35e-3f){
			firing = 1.0f;
		}
	}

	// write back dynamics state
	dynState[DYN_STATE_V * numNeurons + nId] = v;
	dynState[DYN_STATE_I * numNeurons + nId] = Itotal;
	dynState[DYN_STATE_M * numNeurons + nId] = m;
	dynState[DYN_STATE_H * numNeurons + nId] = h;
	dynState[DYN_STATE_N * numNeurons + nId] = n;

	// write firing
	firingVec[nId] = firing;
}

/*
** Benchmarking on a GeForce GTX 580 showed that best performance
** is achieved with 32 threads per warp and 20 warps per block.
** See commit 569c50a3eab78bd089a25d7c04d79a1103279a7e
*/
#define NUM_WARPS 20
int neuronUpdate(
	const float * cond,
	neuron_t * neuron,
	float * firing
){
	// reset CUDA error
	cudaGetLastError();

	// block size
	int blockSize = 32 * NUM_WARPS;

	// update neurons
	dim3 threads(blockSize);
	dim3 grid((int) ceil(
		(double) neuron->numNeurons / blockSize
	));

	// launch kernel
	neuronUpdateKernel<<<grid, threads>>>(
		neuron->numNeurons,
		cond,
		neuron->dynParam,
		neuron->dynState,
		firing
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
