#include <stdio.h>
#include "net.h"

int netNew(
	net_t * this,
	int numNeurons,
	int numSynapses
){
	int error;

	error = neuronNew(&this->neuron, numNeurons);
	if(error) return -1;

	error = synapseNew(&this->synapse, numNeurons, numSynapses);
	if(error) return -1;

	return 0;
}

int netToGPU(net_t * this){
	int error;

	error = neuronToGPU(&this->neuron);
	if(error) return -1;

	error = synapseToGPU(&this->synapse);
	if(error) return -1;

	return 0;
}

int netUpdate(net_t * this){
	int error;

	// update all synapses
	error = synapseUpdate(&this->synapse, &this->neuron);
	if(error){
		printf("Could not update synapses\n");
		return -1;
	}

	// update all neurons
	error = neuronUpdate(&this->neuron);
	if(error){
		printf("Could not update neurons\n");
		return -1;
	}

	return 0;
}

int netRead(
	net_t * this,
	const char * dynParamFile,
	const char * dynStateFile,
	const char * synapseFile
){
	int error;

	error = neuronRead(&this->neuron, dynParamFile, dynStateFile);
	if(error) return -1;

	error = synapseRead(&this->synapse, synapseFile);
	if(error) return -1;

	return 0;
}

int netReadSize(
	int * pNumNeurons,
	int * pSynSuper,
	int * pSynSub,
	const char * dynParamFile,
	const char * dynStateFile,
	const char * synapseFile
){
	int error;

	/*
	** check dynamics parameter matrix
	*/
	int dynParamRows, dynParamCols;
	error = ioReadMatSize(
		dynParamFile,
		&dynParamRows,
		&dynParamCols
	);

	if(error) return error;

/*
** TODO
	if(dynParamCols != DYN_PARAM_LEN){
		printf("Invalid column count in %s\n", dynParamFile);
		return -1;
	}
*/

	// set number of neurons
	const int numNeurons = dynParamRows;

	/*
	** check dynamics state matrix
	*/
	int dynStateRows, dynStateCols;
	error = ioReadMatSize(
		dynStateFile,
		&dynStateRows,
		&dynStateCols
	);

	if(error) return error;

	if(dynStateRows != numNeurons){
		printf("Invalid row count in %s\n", dynStateFile);
		return -1;
	}

/*
** TODO
	if(dynStateCols != DYN_STATE_LEN){
		printf("Invalid column count in %s\n", dynStateFile);
		return -1;
	}
*/

	/*
	** check synapse matrix
	**
	** NOTICE
	** The synapse matrix is assumed to be banded with equal number of
	** superdiagonals and subdiagonals. Together with the main diagonal
	** this makes an odd number of rows in the matrix.
	*/
	int synapseRows, synapseCols;
	error = ioReadMatSize(
		synapseFile,
		&synapseRows,
		&synapseCols
	);

	if(error) return error;

	if(synapseRows % 2 == 0){
		printf("Expected odd number of rows in %s\n", synapseFile);
		return -1;
	}

	if(synapseCols != numNeurons){
		printf("Invalid column count in %s\n", synapseFile);
	}

	const int synSuper = (synapseRows - 1) / 2;
	const int synSub = (synapseRows - 1) / 2;

	// write back
	*pNumNeurons = numNeurons;
	*pSynSuper = synSuper;
	*pSynSub = synSub;

	return 0;
}
