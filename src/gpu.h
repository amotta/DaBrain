#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif

int gpuInit();
int gpuCopyTo(
	const void * hPtr,
	void ** dPtr,
	size_t size
);
int gpuCopyMemoryFromGPU(
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
int gpuMultiplySV(
	int vecRows,
	const float * alpha,
	float * vec
);

#ifdef __cplusplus
}
#endif

#endif
