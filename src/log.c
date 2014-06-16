#include "log.h"

void logFiring(const net_t * pNet, FILE * logFile){
	if(!logFile) return;

	for(int n = 0; n < pNet->numNeurons; n++){
		if(pNet->firing[n]){
			fprintf(logFile, "%d\t%d\n", pNet->t, n);
		}
	}
}
