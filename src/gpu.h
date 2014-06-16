#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif

void gpuCopyMemory(const void * hPtr, void ** dPtr, size_t size);
void gpuMultiplyMV(
	const float * mat,
	int matRows,
	int matCols,
	const float * vecIn,
	int vecInStride,
	float * vecOut,
	int vecOutStride
);
void gpuUpdateState(
	int numNeurons,
	float * dynState,
	const float * dynParam
);

#ifdef __cplusplus
}
#endif

#endif
