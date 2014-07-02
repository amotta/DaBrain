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

	// init GPU
	printf("Init GPGPU... ");
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

	return EXIT_SUCCESS;
}
