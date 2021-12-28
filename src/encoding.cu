#include "../include/encoding.cuh"

__device__ float* get2df(float* p, const int x, int y, const int stride) {
        return (float*)((char*)p + x*stride) + y;
}

__global__ void encodeLevelId(
	float* level_hvs, float* id_hvs, float* feature_matrix, float* hv_matrix,
    int level_stride, int id_stride, int fm_stride, int N, int F, int Q, int D)
{
    const int sample_idx = blockIdx.y;
	if (sample_idx >= N)
        return;

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
	if (d >= D)
        return;

	int f;
    float encoded_hv_e = 0.0;
#pragma unroll 1
    for (f = 0; f < F; ++f) {
        float v = *get2df(feature_matrix, sample_idx, f, fm_stride);
        encoded_hv_e += *get2df(level_hvs, (int)(v * Q), d, level_stride) * \
                        *get2df(id_hvs, f, d, id_stride);
    }

    hv_matrix[sample_idx * D + d] = encoded_hv_e;
}

__global__ void encodeIDNgram(float* hv_matrix, float* level_hvs, float* id_hvs, float* feature_matrix,
                            int level_stride, int id_stride, int fm_stride, const int Q, const int F, 
                            const int ngramN, const int dataN, const int D) {
    const int sample_idx = blockIdx.y;
    if (sample_idx >= dataN)
        return;

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
	if (d >= D)
        return;
    
    float encoded_hv_e = 0.0;
    #pragma unroll 1
    for (int f = 0; f < F; ++f) {
        float v = *get2df(feature_matrix, sample_idx, f, fm_stride);
        for(int k = 0; k < ngramN; ++k) {
            if (d + k < D) {
                encoded_hv_e += *get2df(level_hvs, (int)(v * Q), d + k, level_stride) * \
                                *get2df(id_hvs, f, d + k, id_stride);
            }
            else {
                encoded_hv_e += *get2df(level_hvs, (int)(v * Q), d + k - D, level_stride) * \
                                *get2df(id_hvs, f, d + k - D, id_stride);
            }
        }
    }
    hv_matrix[sample_idx * D + d] = encoded_hv_e;
}
