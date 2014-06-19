#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpu.h"
#include "log.h"
#include "net.h"

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

	printf("done\n");
	printf(
		"> Number of neurons: %d\n"
		"> Number of synapses per neuron: %d\n",
		numNeurons,
		synSuper + 1 + synSub
	);

	// create new network
	net_t net = {
		.numNeurons = numNeurons,
		.synSuper = synSuper,
		.synSub = synSub
	};
	netNew(&net);

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

		// copy firing neurons to host
		gpuCopyMemoryFromGPU(
			gpuNet.firing,
			net.firing,
			net.numNeurons * sizeof(float)
		);
		logFiring(&net, firingFile);
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

	return EXIT_SUCCESS;
}
