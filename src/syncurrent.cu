#include <stdio.h>
#include "gpu.h"
#include "syncurrent.h"

int synUpdate(synapse_t * syn, neuron_t * neuron);

const synapse_t synCurrent = {
	.name ="Instant synaptic current",
	.update = synUpdate
};

int synUpdate(synapse_t * syn, neuron_t * neuron){
	int numSuperDiag = (syn->numSynapses - 1) / 2;
	int numSubDiag = (syn->numSynapses - 1) / 2;

	int error = gpuMultiplyBMV(
		// synapse matrix
		syn->mat,
		// dimensions
		syn->numNeurons,
		syn->numNeurons,
		// number of diagonals		
		numSuperDiag,
		numSubDiag,
		// firing vector
		neuron->fire, 1,
		neuron->iSyn, 1
	);

	if(error){
		printf("Could not update current\n");
		return -1;
	}

	return 0;
}
