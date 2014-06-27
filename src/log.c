#include "log.h"

void logFiring(const net_t * pNet, FILE * logFile){
	if(!logFile) return;

	int offset = DYN_STATE_V * pNet->numNeurons; 
	float * vArr = &pNet->dynState[offset];

	for(int n = 0; n < pNet->numNeurons; n++){
		if(vArr[n] > -35.0e-3f){
			fprintf(logFile, "%d\t%d\n", pNet->t, n);
		}
	}
}

void logCurrent(const net_t * pNet, FILE * logFile){
	if(!logFile) return;

	int offset = DYN_STATE_I * pNet->numNeurons;
	float * iArr = &pNet->dynState[offset];

	// log time
	fprintf(logFile, "%d", pNet->t);

	// log currents
	for(int n = 0; n < pNet->numNeurons; n++){
		fprintf(logFile, " %e", iArr[n]);
	}

	// end line
	fprintf(logFile, "\n");
}