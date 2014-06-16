#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif

void gpuCopyMemory(const void * hPtr, void ** dPtr, size_t size);
void gpuMultiplyMV(
	const float * mat,
	const float * vectIn,
	float * vecOut,
	int rows, int cols
);

#ifdef __cplusplus
}
#endif

#endif