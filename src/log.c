#include <stdio.h>
#include "log.h"

int logVector(
	const int vecLen,
	const float * vec,
	FILE * logFile
){
	if(!logFile){
		printf("Invalid log file\n");
		return -1;
	}

	for(int n = 0; n < vecLen; n++){
		if(n > 0){
			fprintf(logFile, " ");
		}

		fprintf(logFile, "%e", vec[n]);
	}

	return 0;
}