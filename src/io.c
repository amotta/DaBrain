#include <stdio.h>
#include <string.h>
#include "io.h"
#include "io_bin.h"
#include "io_csv.h"

enum FILE_TYPE {
	FILE_TYPE_BIN,
	FILE_TYPE_CSV,
	FILE_TYPE_ERR
};

int fileNameToType(const char * fileName){
	const char * suffix = strchr(fileName, '.');

	if(!suffix){
		printf("File name has no suffix\n");
		return FILE_TYPE_ERR;
	}

	if(!strcmp(".bin", suffix)) return FILE_TYPE_BIN;
	if(!strcmp(".csv", suffix)) return FILE_TYPE_CSV;

	printf("File name has invalid suffix\n");
	return FILE_TYPE_ERR;
}

int ioReadMat(
	const char * fileName,
	const int rows,
	const int cols,
	float * mat
){
	int error = 0;
	int type = fileNameToType(fileName);

	switch(type){
		case FILE_TYPE_BIN:
			error = ioBinReadMat(
				fileName,
				rows, cols,
				mat
			);
			break;

		case FILE_TYPE_CSV:
			error = ioCsvReadMat(
				fileName,
				rows, cols,
				mat
			);
			break;

		default:
			error = -1;
			break;
	}

	return error;
}

int ioReadMatSize(
	const char * fileName,
	int * rows,
	int * cols
){
	int error = 0;
	int type = fileNameToType(fileName);

	switch(type){
		case FILE_TYPE_BIN:
			error = ioBinReadMatSize(
				fileName,
				rows, cols
			);
			break;

		case FILE_TYPE_CSV:
			error = ioCsvReadMatSize(
				fileName,
				rows, cols
			);
			break;

		default:
			error = -1;
			break;
	}

	return error;
}
