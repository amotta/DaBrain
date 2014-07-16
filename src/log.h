#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include "net.h"

int logVector(
	const int vecLen,
	const float * vec,
	FILE * logFile
);

#endif
