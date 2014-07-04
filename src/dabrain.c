#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "net.h"
#include "goldman.h"
#include "syncurrent.h"

static const char * dynParamFile = "dynParam.csv";
static const char * dynStateFile = "dynState.csv";
static const char * synFile = "syn.csv";

int main(int argc, char ** argv){
	int error;

	// check CSV files
	printf("Analyse CSV files... ");
	fflush(stdout);

	int numNeurons;
	int synSuper;
	int synSub;
	error = netReadSize(
		&numNeurons,
		&synSuper,
		&synSub,
		dynParamFile,
		dynStateFile,
		synFile
	);
	if(error) return EXIT_FAILURE;

	int numSynapses = synSuper + 1 + synSub;

	printf("done\n");
	printf(
		"> Number of neurons: %d\n"
		"> Number of synapses per neuron: %d\n",
		numNeurons,
		numSynapses
	);

	// choose neuron and synapse model
	neuron_t neuron = neuronGold;
	synapse_t synapse = synCurrent;

	// configure network
	net_t net = {
		.neuron = neuron,
		.synapse = synapse
	};

	// allocate memory
	netNew(&net, numNeurons, numSynapses);

	// load network from CSV files
	printf("Loading CSV files... ");
	fflush(stdout);

	error = netRead(
		&net,
		dynParamFile,
		dynStateFile,
		synFile
	);
	if(error) return EXIT_FAILURE;

	printf("done\n");

	printf("Check... ");
	fflush(stdout);

	for(int i = 0; i < 100; i++){
		printf(
			"Neuron %d: %e\n",
			i,
			net.neuron.param[i]
		);
	}

#if 0
	// init GPU
	printf("Initizaling GPU... ");
	fflush(stdout);

	gpuInit();

	printf("done\n");

	// copy to GPU
	printf("Copying data to GPU... ");
	fflush(stdout);

	net_t gpuNet = net;
	error = netToGPU(&gpuNet);
	if(error){
		printf("Failed to copy network to GPU.\n");
		return EXIT_FAILURE;
	}

	printf("done\n");

	// prepare logging
	FILE * firingFile;
	firingFile = fopen("firing.log", "w");

	FILE * currentFile;
	currentFile = fopen("current.log", "w");

	// prepare benchmark
	clock_t tic;
	clock_t toc;

	// start simulation
	printf("Running simulation... ");
	fflush(stdout);

	tic = clock();
	while(net.t < 1000){
		error = netUpdate(&gpuNet);
		if(error){
			printf("Error while updating network.\n");
			printf("Abort simulation.\n");
			break;
		}

		// update clock
		net.t++;
		gpuNet.t++;

		/*
		** logging
		** Only sample with 200 Hz in order to increase execution speed.
		** The Nyquist frequency is still high enough to observe the
		** band of gamma frequencies.
		*/
		if(net.t % 5 == 0){
			// copy firing neurons to host
			gpuCopyMemoryFromGPU(
				gpuNet.dynState,
				net.dynState,
				2 * net.numNeurons * sizeof(float)
			);
			
			logFiring(&net, firingFile);
			logCurrent(&net, currentFile);
		}
	}
	toc = clock();

	// show stats
	printf("done\n");
	printf(
		"> Duration: %f\n",
		(float) (toc - tic) / CLOCKS_PER_SEC
	);

	// end logging
	fclose(firingFile);
	fclose(currentFile);
#endif

	return EXIT_SUCCESS;
}
