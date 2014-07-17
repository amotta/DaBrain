#include <stdio.h>
#include "log.h"

int logVector(
	const int vecLen,
	const float * vec,
	FILE * logFile
){
	if(!logFile) return -1;

	for(int n = 0; n < vecLen; n++){
		// delimiter
		if(n) fprintf(logFile, " ");

		// value
		fprintf(logFile, "%e", vec[n]);
	}

	// line break
	fprintf(logFile, "\n");

	return 0;
}

int logVectorStamped(
	const int stamp,
	const int vecLen,
	const float * vec,
	FILE * logFile
){
	int error;

	if(!logFile) return -1;

	// stamp
	fprintf(logFile, "%d ", stamp);

	// vector
	error = logVector(vecLen, vec, logFile);
	if(error) return -1;

	return 0;
}
