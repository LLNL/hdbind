#ifndef __ENCODING_CUH__
#define __ENCODING_CUH__

__device__ float* get2df(float* p, const int x, int y, const int stride);
__global__ void encodeLevelId(
	float* level_hvs, float* id_hvs, float* feature_matrix, float* hv_matrix,
    int level_stride, int id_stride, int fm_stride, int N, int F, int Q, int D);
__global__ void encodeIDNgram(float* hv_matrix, float* level_hvs, float* id_hvs, float* feature_matrix,
        int level_stride, int id_stride, int fm_stride, const int Q, const int F, 
        const int ngramN, const int dataN, const int D);

#endif
