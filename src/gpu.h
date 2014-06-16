#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif

void gpuCopyMemory(
	const void * hPtr,
	void ** dPtr,
	size_t size
);

#ifdef __cplusplus
}
#endif

#endif