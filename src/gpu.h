#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif

int gpuInit();
int gpuCopyTo(
	const size_t size,
	const void * hPtr,
	void ** dPtr
);
int gpuCopyFrom(
	const size_t size,
	const void * dPtr,
	void * hPtr
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
