#ifndef IO_CSV_H
#define IO_CSV_H

int ioCsvReadMat(
	const char * fileName,
	float * mat,
	int rows,
	int cols
);
int ioCsvReadMatSize(
	const char * fileName,
	int * rows,
	int * cols
);

#endif
