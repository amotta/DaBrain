#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif

void gpuCopyMemoryToGPU(const void * hPtr, void ** dPtr, size_t size);
void gpuCopyMemoryFromGPU(const void * dPtr, void * hPtr, size_t size);
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
	float * firing,
	const float * dynParam,
	const float * Isyn
);

#ifdef __cplusplus
}
#endif

#endif
