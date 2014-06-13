#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "log.h"
#include "net.h"

void usage(){
	printf("dabrain [options] numNeurons\n");
	printf("\tNo options yet\n");
}

int main(int argc, char ** argv){
	if(argc < 2){
		usage();
		return EXIT_SUCCESS;
	}

	size_t numNeurons = strtol(argv[1], NULL, 10);
	net_t net = {
		.numNeurons = numNeurons
	};

	netNew(&net);
	netInit(&net);
	printf("Net build\n");
	printf("Size: %lu\n", net.numNeurons);

	// prepare logging
	FILE * firingFile = fopen("firing.log", "w");
	
	clock_t tic = clock();	
	for(size_t i = 0; i < 1000; i++){
		logFiring(&net, firingFile);
		netUpdate(&net);
	}
	clock_t toc = clock();

	// end logging
	fclose(firingFile);

	printf("Net updated\n");
	printf("Time elapsed: %f\n", (double) (toc - tic) / CLOCKS_PER_SEC);	
	return EXIT_SUCCESS;
}
