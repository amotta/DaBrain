#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpu.h"
#include "log.h"
#include "net.h"

int main(int argc, char ** argv){
	int error;

	// check files 
	printf("Analyse files... ");
	fflush(stdout);

	int numNeurons;
	int numSyn;
	error = netReadSize(&numNeurons, &numSyn);
	if(error) return EXIT_FAILURE;

	printf("done\n");
	printf(
		"> Number of neurons: %d\n"
		"> Number of synapses per neuron: %d\n",
		numNeurons,
		numSyn
	);

	// create network
	net_t net = {
		.numNeurons = numNeurons,
		.neurons = {
			.numNeurons = numNeurons
		},
		.syn = {
			.numNeurons = numNeurons,
			.numSyn = numSyn
		}
	};

	// allocate memory
	netNew(&net);

	// load network from files
	printf("Loading files... ");
	fflush(stdout);

	error = netRead(&net);
	if(error) return EXIT_FAILURE;

	printf("done\n");

	// init GPU
	printf("Initizaling GPU... ");
	fflush(stdout);

	gpuInit();

	printf("done\n");

	// copy to GPU
	printf("Copying data to GPU... ");
	fflush(stdout);

	net_t gpuNet = net;
	error = netCopyToGPU(&gpuNet);
	
	if(error){
		printf("Failed to copy network to GPU.\n");
		return EXIT_FAILURE;
	}

	printf("done\n");

	// prepare logging
	FILE * firingFile;
	firingFile = fopen("firing.log", "w");

	// prepare benchmark
	clock_t tic;
	clock_t toc;

	// start simulation
	printf("Running simulation... ");
	fflush(stdout);

	// start benchmarking
	tic = clock();

	for(int t = 0; t < 1000; t++){
		error = netUpdate(&gpuNet);

		if(error){
			printf("Error during network update\n");
			printf("Abort simulation.\n");
			break;
		}

		// copy voltages to host
		error = gpuCopyFrom(
			net.numNeurons * sizeof(float),
			gpuNet.neurons.dynState,
			net.neurons.dynState
		);

		if(error){
			printf("Failed to copy data to host\n");
			return -1;
		}

		// log voltages
		logVectorStamped(
			t,
			net.numNeurons,
			net.neurons.dynState,
			firingFile
		);
	}

	// stop benchmarking
	toc = clock();

	// show stats
	printf("done\n");
	printf(
		"> Duration: %f\n",
		(float) (toc - tic) / CLOCKS_PER_SEC
	);

	// end logging
	fclose(firingFile);

	return EXIT_SUCCESS;
}
