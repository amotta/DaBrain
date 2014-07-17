#include <stdio.h>
#include <string.h>
#include "io.h"

int ioReadMat(
	const char * fileName,
	const int rows,
	const int cols,
	float * mat
){
	// open file
	FILE * file = fopen(fileName, "r");

	// check for error
	if(!file){
		printf("Could not open file %s\n", fileName);
		return -1;
	}

	// skip the matrix size
	int error = fseek(
		file,
		2 * sizeof(int),
		SEEK_CUR
	);

	if(error){
		printf("Could not skip matrix size\n");

		fclose(file);
		return -1;
	}

	size_t numTotal = (size_t) rows * cols;
	size_t numRead = fread(
		(void *) mat,
		sizeof(float),
		numTotal,
		file
	);

	if(numRead < numTotal){
		printf("Failed to read file %s\n", fileName),

		fclose(file);
		return -1;
	}

	// close file
	fclose(file);
	return 0;
}

int ioReadMatSize(
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