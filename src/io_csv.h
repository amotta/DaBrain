#ifndef IO_CSV_H
#define IO_CSV_H

int ioCsvReadMat(
	const char * fileName,
	const int rows,
	const int cols,
	float * mat
);

int ioCsvReadMatSize(
	const char * fileName,
	int * rows,
	int * cols
);

#endif
