#ifndef IO_H
#define IO_H

#ifdef __cplusplus
extern "C" {
#endif

int ioReadMat(
	const char * fileName,
	const int rows,
	const int cols,
	float * mat
);

int ioReadMatSize(
	const char * fileName,
	int * rows,
	int * cols
);

#ifdef __cplusplus
}
#endif

#endif
