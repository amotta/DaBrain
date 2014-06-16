#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpu.h"
#include "log.h"
#include "net.h"

void usage(){
	printf("dabrain [options] numNeurons\n");
}

int main(int argc, char ** argv){
	if(argc < 2){
		usage();
		return EXIT_SUCCESS;
	}

	// read number of neurons
	int numNeurons = (int) strtol(argv[1], NULL, 10);
	int numExc = (int) (0.8 * numNeurons);

	net_t net = {
		.numNeurons = numNeurons,
		.numExc = numExc
	};

	netNew(&net);
	netInit(&net);
	printf("Net build\n");
	printf("Size: %d\n", net.numNeurons);

	// copy to gpu
	net_t dNet = netCopyToGPU(&net);

	// prepare logging
	FILE * firingFile = fopen("firing.log", "w");
	
	clock_t tic = clock();

	while(dNet.t < 1000){
		netUpdate(&dNet);

		// update clock
		net.t++;
		dNet.t++;

		// copy firing neurons to host
		gpuCopyMemoryFromGPU(
			dNet.firing,
			net.firing,
			(size_t) net.numNeurons * sizeof(float)
		);
		logFiring(&net, firingFile);
	}

	clock_t toc = clock();

	// end logging
	fclose(firingFile);

	printf("Net updated\n");
	printf("Time elapsed: %f\n", (double) (toc - tic) / CLOCKS_PER_SEC);	
	return EXIT_SUCCESS;
}
