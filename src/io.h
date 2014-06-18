#ifndef IO_H
#define IO_H

int ioReadMat(const char * fileName, float * mat, int rows, int cols);
int ioReadMatSize(const char * fileName, int * rows, int * cols);

#endif
