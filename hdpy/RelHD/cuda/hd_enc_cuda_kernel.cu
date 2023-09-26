#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define MIN(x, y) ((x < y) ? x : y)
#define PACK_UNIT_SIZE 32


__global__ void packing_cuda_kernel(torch::PackedTensorAccessor<int,2,torch::RestrictPtrTraits, size_t> output, 
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> arr, 
                                    int origLength, int packLength, int numVec)
                    {
                        int i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= origLength)
                            return;
                        for (int sample_idx = blockIdx.y; sample_idx < numVec; sample_idx += blockDim.y * gridDim.y) 
                        {
                            int tid = threadIdx.x;
                            int lane = tid % warpSize;
                            int bitPattern=0;
                            if (i < origLength)
                                bitPattern = __brev(__ballot_sync(0xFFFFFFFF, arr[sample_idx][i] > 0));
                                // bitPattern = __brev(__ballot_sync(0xFFFFFFFF, arr[sample_idx*origLength+i] > 0));
                            if (lane == 0) {
                                output[sample_idx][(i / warpSize)] = bitPattern;
                                // output[sample_idx*packLength+ (i / warpSize)] = bitPattern;
                            }
                        }
                    }


__global__ void dense_idlv_cuda_kernel(
                                    torch::PackedTensorAccessor<int,2,torch::RestrictPtrTraits, size_t> level_hvs_packed,
                                    torch::PackedTensorAccessor<int,2,torch::RestrictPtrTraits, size_t> id_hvs_packed,
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> hv_matrix,
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> feature_matrix,
                                    int N, int Q, int F, int D) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) {

        float encoded_hv_e = 0.0;
        #pragma unroll 1
        for (int f = 0; f < F; ++f) {
            float v = feature_matrix[sample_idx][f];
            int id_hv_val = ((((id_hvs_packed[f][d/32]) >> (31 - d % 32)) & 0x01) == 0.) ? -1 : 1;
            int lv_hv_val = ((((level_hvs_packed[(int)(v * Q)][d/32]) >> (31 - d % 32)) & 0x01) == 0.) ? -1 : 1;

            encoded_hv_e += id_hv_val * lv_hv_val;
        }

        hv_matrix[sample_idx][d] = encoded_hv_e;
    }
}

__global__ void hd_enc_lvid_cuda_kernel(
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> level_hvs,
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> id_hvs,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> feature_indices,
                                    torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits, size_t> feature_values,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> csr_info,
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> hv_matrix,
                                    int N, int Q, int D) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) {
        float encoded_hv_e = 0.0;
        unsigned int start_range = csr_info[sample_idx];
        unsigned int end_range = csr_info[sample_idx + 1];

        #pragma unroll 1
        for (int f = start_range; f < end_range; ++f) {
            encoded_hv_e += level_hvs[(int)(feature_values[f] * Q)][d] * id_hvs[feature_indices[f]][d];
        }

        hv_matrix[sample_idx][d] = encoded_hv_e;
    }
}
__global__ void hd_enc_id_cuda_kernel(
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> id_hvs,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> feature_indices,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> csr_info,
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> hv_matrix,
                                    int N, int D) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) {
        float encoded_hv_e = 0.0;
        unsigned int start_range = csr_info[sample_idx];
        unsigned int end_range = csr_info[sample_idx + 1];

        #pragma unroll 1
        for (int f = start_range; f < end_range; ++f) {
            encoded_hv_e += id_hvs[feature_indices[f]][d];
        }

        hv_matrix[sample_idx][d] = encoded_hv_e;
    }
}
__global__ void hd_enc_id_packed_cuda_kernel(
                                    torch::PackedTensorAccessor<unsigned char,2,torch::RestrictPtrTraits, size_t> id_hvs_packed,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> feature_indices,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> csr_info,
                                    torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> hv_matrix,
                                    int N, int D) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) {
        float encoded_hv_e = 0.0;
        unsigned int start_range = csr_info[sample_idx];
        unsigned int end_range = csr_info[sample_idx + 1];

        #pragma unroll 1
        for (int f = start_range; f < end_range; ++f) {
            int v = ((((id_hvs_packed[feature_indices[f]][d/8]) >> (7 - d % 8)) & 0x01) == 0.) ? -1 : 1;
            encoded_hv_e += v;
        }

        hv_matrix[sample_idx][d] = encoded_hv_e;
    }
}


__global__ void memory_hv_enc_cuda_kernel(
                                    torch::PackedTensorAccessor<short,2,torch::RestrictPtrTraits, size_t> node_hvs,
                                    // torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> node_hvs,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> idx,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> idx_ptr,
                                    torch::PackedTensorAccessor<int,2,torch::RestrictPtrTraits, size_t> memory_hv_matrix,
                                    // torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits, size_t> memory_hv_matrix,
                                    int N, int D) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) {
        int acc_e = 0;
        unsigned int start_range = idx_ptr[sample_idx];
        unsigned int end_range = idx_ptr[sample_idx + 1];

        #pragma unroll 1
        for (int f = start_range; f < end_range; ++f) {
            acc_e += node_hvs[idx[f]][d];
        }

        memory_hv_matrix[sample_idx][d] = acc_e;
        // memory_hv_matrix[sample_idx][d] = (float) acc_e;
    }
}

__global__ void memory_hv_lv2_enc_cuda_kernel(
                                    torch::PackedTensorAccessor<int,2,torch::RestrictPtrTraits, size_t> node_hvs,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> idx,
                                    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits, size_t> idx_ptr,
                                    torch::PackedTensorAccessor<int,2,torch::RestrictPtrTraits, size_t> memory_hv_matrix,
                                    int N, int D) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) {
        int acc_e = 0;
        unsigned int start_range = idx_ptr[sample_idx];
        unsigned int end_range = idx_ptr[sample_idx + 1];

        #pragma unroll 1
        for (int f = start_range; f < end_range; ++f) {
            acc_e += node_hvs[idx[f]][d];
        }

        memory_hv_matrix[sample_idx][d] = acc_e;
    }
}


torch::Tensor packing_cuda(
    torch::Tensor orig_vec,
    const int N, const int D) {

    auto options = torch::TensorOptions()
                    .dtype(torch::kInt32)
                    .device(torch::kCUDA, 0);
    const int packed_dim = (D+PACK_UNIT_SIZE-1)/PACK_UNIT_SIZE;
    auto packed_vec = torch::zeros({N, packed_dim}, options);

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    const int threads = 1024;
    const dim3 blocks((D + threads - 1) / threads, MIN(N, prop.maxGridSize[1]));
    cudaError_t err;
    packing_cuda_kernel<<<blocks, threads>>>(packed_vec.packed_accessor<int,2,torch::RestrictPtrTraits, size_t>(),
                                                orig_vec.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                D, packed_dim, N);


    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return packed_vec;
}


torch::Tensor dense_idlv_cuda(
    torch::Tensor level_hvs_packed,
    torch::Tensor id_hvs_packed,
    torch::Tensor raw_data,
    const int N, const int Q, const int F, const int D) {

    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCUDA, 0);

    auto encoded_hvs = torch::zeros({N, D}, options);

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    const int threads = 1024;
    const dim3 blocks((D + threads - 1) / threads, MIN(N, prop.maxGridSize[1]));
    cudaError_t err;
    dense_idlv_cuda_kernel<<<blocks, threads>>>(level_hvs_packed.packed_accessor<int,2,torch::RestrictPtrTraits, size_t>(),
                                                id_hvs_packed.packed_accessor<int,2,torch::RestrictPtrTraits, size_t>(),
                                                encoded_hvs.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                raw_data.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                N, Q, F, D);


    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return encoded_hvs;
}


torch::Tensor hd_enc_lvid_cuda(
    torch::Tensor level_hvs,
    torch::Tensor id_hvs,
    torch::Tensor feature_indices,
    torch::Tensor feature_values,
    torch::Tensor csr_info,
    const int N, const int Q, const int D) {

    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCUDA, 0);

    auto hv_matrix = torch::zeros({N, D}, options);

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    const int threads = 1024;
    const dim3 blocks((D + threads - 1) / threads, MIN(N, prop.maxGridSize[1]));
    cudaError_t err;
    hd_enc_lvid_cuda_kernel<<<blocks, threads>>>(level_hvs.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                id_hvs.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                feature_indices.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                feature_values.packed_accessor<float,1,torch::RestrictPtrTraits, size_t>(),
                                                csr_info.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                hv_matrix.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                N, Q, D);


    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return hv_matrix;
}

torch::Tensor hd_enc_id_cuda(
    torch::Tensor id_hvs,
    torch::Tensor feature_indices,
    torch::Tensor csr_info,
    const int N, const int D) {

    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCUDA, 0);

    auto hv_matrix = torch::zeros({N, D}, options);

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    const int threads = 1024;
    const dim3 blocks((D + threads - 1) / threads, MIN(N, prop.maxGridSize[1]));
    cudaError_t err;
    hd_enc_id_cuda_kernel<<<blocks, threads>>>(id_hvs.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                feature_indices.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                csr_info.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                hv_matrix.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                N, D);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return hv_matrix;
}


torch::Tensor hd_enc_id_packed_cuda(
    torch::Tensor id_hvs_packed,
    torch::Tensor feature_indices,
    torch::Tensor csr_info,
    const int N, const int D) {

    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCUDA, 0);

    auto hv_matrix = torch::zeros({N, D}, options);

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    const int threads = 1024;
    const dim3 blocks((D + threads - 1) / threads, MIN(N, prop.maxGridSize[1]));
    cudaError_t err;
    hd_enc_id_packed_cuda_kernel<<<blocks, threads>>>(id_hvs_packed.packed_accessor<unsigned char,2,torch::RestrictPtrTraits, size_t>(),
                                                    feature_indices.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                    csr_info.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                    hv_matrix.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                    N, D);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return hv_matrix;
}

torch::Tensor memory_hv_enc_cuda(
    torch::Tensor node_hvs,
    torch::Tensor idx,
    torch::Tensor idx_ptr,
    const int N, const int D) {

    auto options = torch::TensorOptions()
                    .dtype(torch::kInt32)
                    // .dtype(torch::kFloat32)
                    .device(torch::kCUDA, 0);

    auto memory_hv_matrix = torch::zeros({N, D}, options);

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    const int threads = 1024;
    const dim3 blocks((D + threads - 1) / threads, MIN(N, prop.maxGridSize[1]));
    cudaError_t err;
    memory_hv_enc_cuda_kernel<<<blocks, threads>>>(node_hvs.packed_accessor<short,2,torch::RestrictPtrTraits, size_t>(),
    // memory_hv_enc_cuda_kernel<<<blocks, threads>>>(node_hvs.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                    idx.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                    idx_ptr.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                    memory_hv_matrix.packed_accessor<int,2,torch::RestrictPtrTraits, size_t>(),
                                                    // memory_hv_matrix.packed_accessor<float,2,torch::RestrictPtrTraits, size_t>(),
                                                    N, D);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return memory_hv_matrix;
}

torch::Tensor memory_hv_lv2_enc_cuda(
    torch::Tensor node_hvs,
    torch::Tensor idx,
    torch::Tensor idx_ptr,
    const int N, const int D) {

    auto options = torch::TensorOptions()
                    .dtype(torch::kInt32)
                    // .dtype(torch::kFloat32)
                    .device(torch::kCUDA, 0);

    auto memory_hv_matrix = torch::zeros({N, D}, options);

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    const int threads = 1024;
    const dim3 blocks((D + threads - 1) / threads, MIN(N, prop.maxGridSize[1]));
    cudaError_t err;
    memory_hv_lv2_enc_cuda_kernel<<<blocks, threads>>>(node_hvs.packed_accessor<int,2,torch::RestrictPtrTraits, size_t>(),
                                                    idx.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                    idx_ptr.packed_accessor<int,1,torch::RestrictPtrTraits, size_t>(),
                                                    memory_hv_matrix.packed_accessor<int,2,torch::RestrictPtrTraits, size_t>(),
                                                    N, D);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return memory_hv_matrix;
}