#ifndef IO_BIN_H
#define IO_BIN_H

int ioBinReadMat(
	const char * fileName,
	const int rows,
	const int cols,
	float * mat
);

int ioBinReadMatSize(
	const char * fileName,
	int * rows,
	int * cols
);

#endif
