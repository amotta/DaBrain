#include "log.h"

void logFiring(net_t * pNet, FILE * logFile){
	if(!logFile) return;

	for(int n = 0; n < pNet->numNeurons; n++){
		float * state = &pNet->dynState[n * DYN_STATE_LEN];
		if(state[DYN_STATE_FIRING]){
			fprintf(logFile, "%d\t%d\n", pNet->t, n);
		}
	}
}
