#include <stdio.h>
#include "io_bin.h"

#define BUF_SIZE 256

int ioBinReadMat(
	const char * fileName,
	const int rows,
	const int cols,
	float * mat
){
	FILE * file = fopen(fileName, "r");

	// check for error
	if(!file){
		printf("Could not open file %s\n", fileName);
		return -1;
	}

	int status = 0;
	size_t numDone = 0;
	size_t numTotal = (size_t) rows * cols;

	while(numDone < numTotal){
		size_t numRead = fread(
			(void *) &mat[numDone],
			sizeof(float),
			BUF_SIZE,
			file
		);

		if(!numRead){
			printf("Error while reading file %s\n", fileName);
			status = -1;
			break;
		}

		// go to next buffer
		numDone += numRead;
	}

	if(numDone < numTotal){
		printf("Failed to read file %s\n", fileName),
		status = -1;
	}

	// close file
	fclose(file);
	return status;
}

int ioBinReadMatSize(
	const char * fileName,
	int * rows,
	int * cols
){
	FILE * file = fopen(fileName, "r");

	// check for error
	if(!file){
		printf("Could not open file %s\n", fileName);
		return -1;
	}

	int status = 0;
	size_t numRead;

	// read row count
	int numRowsRead;
	numRead = fread((void *) &numRowsRead, sizeof(int), 1, file);

	if(!numRead){
		printf("Could not read row count from %s\n", fileName);

		fclose(file);
		return -1;
	}

	// read column count
	int numColsRead;
	numRead = fread((void *) &numColsRead, sizeof(int), 1, file);

	if(!numRead){
		printf("Could not read column count from %s\n", fileName);

		fclose(file);
		return -1;
	}

	// write back
	*rows = numRowsRead;
	*cols = numColsRead;

	fclose(file);
	return 0;
}
