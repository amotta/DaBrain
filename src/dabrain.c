#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "net.h"

void usage(){
	printf("Number of neurons missing\n");
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
	
	clock_t tic = clock();	
	for(size_t i = 0; i < 1000; i++){
		netUpdate(&net);
	}
	clock_t toc = clock();

	printf("Net updated\n");
	printf("Time elapsed: %f\n", (double) (toc - tic) / CLOCKS_PER_SEC);	
	return EXIT_SUCCESS;
}
