#include "log.h"

void logFiring(net_t * pNet, FILE * logFile){
	if(!logFile) return;

	for(size_t n = 0; n < pNet->numNeurons; n++){
		if(pNet->neurons[n].dynState.firing){
			fprintf(logFile, "%lu\t%lu\n", pNet->time, n);
		}
	}
}
