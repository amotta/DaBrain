#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif

int gpuInit();
void gpuCopyMemoryToGPU(
	const void * hPtr,
	void ** dPtr,
	size_t size
);
void gpuCopyMemoryFromGPU(
	const void * dPtr,
	void * hPtr,
	size_t size
);
int gpuMultiplyBMV(
	const float * mat,
	int matRows,
	int matCols,
	int matSuper,
	int matSub,
	const float * vecIn,
	int vecInStride,
	float * vecOut,
	int vecOutStride
);
int gpuMultiplyMV(
	const float * mat,
	int matRows,
	int matCols,
	const float * vecIn,
	int vecInStride,
	float * vecOut,
	int vecOutStride
);
int gpuScaleV(
	int vecRows,
	const float * alpha,
	float * vec
);

#ifdef __cplusplus
}
#endif

#endif
