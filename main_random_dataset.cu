#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include "include/preprocessor.hpp"
#include "include/cudadebug.cuh"
#include "include/encoding.cuh"
#include "include/kernels.cuh"
#include <cublas_v2.h>
// #include <cuda_runtime.h>

// #define USE_DOT_SIMILARITY
#define USE_COS_SIMILARITY

// #define USE_DOT_ENCODING
#define USE_LVID_ENCODING

__device__ int d_guess;

__global__ void findargmax(float* arr, int offset) {
    float max_val = arr[0];
    int max_idx = 0;
    for (int jj = 1; jj < offset; jj++) {
        if (max_val < arr[jj]) {
            max_val = arr[jj];
            max_idx = jj;
        }
    }
    d_guess = max_idx;
}

void fill_row_int(std::vector<int> & row)
{
    std::generate(row.begin(), row.end(), [](){ return rand() % 100; }); 
}


void fill_row(std::vector<float> & row)
{
    std::generate(row.begin(), row.end(), [](){ return rand() % 100; }); 
}

void fill_matrix(std::vector<std::vector<float>> & mat)
{
    std::for_each(mat.begin(), mat.end(), fill_row);
}

int main(int argc, char* argv[]) {
    // ./main [TRAIN dataset path] [TEST dataset path] [DIM] [ITER] [Learning Rate]
    // Example:
    // ./main datasets/UCIHAR/UCIHAR_train.choir_dat datasets/UCIHAR/UCIHAR_test.choir_dat 10000 20 1
    int nFeatures_train, nClasses_train;  // nFeatures is same as x_train[0].size()
    int nFeatures_test, nClasses_test;
    
    nFeatures_train = atoi(argv[1]);
    nFeatures_test = nFeatures_train;

    nClasses_train = atoi(argv[2]);
    nClasses_test = nClasses_train;

    int train_size = atoi(argv[3]);
    int test_size = atoi(argv[4]);

    std::vector<std::vector<float>> x_test(test_size, std::vector<float>(nFeatures_test));
    std::vector<std::vector<float>> x_train(train_size, std::vector<float>(nFeatures_train));
    std::vector<int> y_train(train_size);
    std::vector<int> y_test(test_size);


    // fill_matrix(x_test);
    // fill_matrix(x_train);
    // fill_row_int(y_test);
    // fill_row_int(y_train);

    // readChoirDat(argv[1], nFeatures_train, nClasses_train, x_train, y_train);
    // readChoirDat(argv[2], nFeatures_test, nClasses_test, x_test, y_test);

    // normalize
    l2norm(x_train);
    l2norm(x_test);

    std::vector<float> x_train_flat = flatten(x_train);
    std::vector<float> x_test_flat = flatten(x_test);

    // base_creation: linear
    int dim = atoi(argv[5]);
    int iter_num = 20;

    float learning_rate = 1; // TODO: Parameterize
    float neg_learning_rate = -learning_rate;

    int Q = 10;

    int train_set_num = x_train.size();
    int test_set_num = x_test.size();
    int base_size = nFeatures_train * dim;
    int train_encode_size = train_set_num * dim;
    int test_encode_size = test_set_num * dim;

    // generate bases
    std::vector<float> bases;  // flattened
    std::vector<float> base_v1(dim/2, 1);
    std::vector<float> base_v2(dim/2, -1);
    base_v1.insert(base_v1.end(), base_v2.begin(), base_v2.end());
    // obtain a time-based seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    for(int i = 0 ; i < nFeatures_train; i++) {
        std::shuffle(base_v1.begin(), base_v1.end(), std::default_random_engine(seed));
        bases.insert(bases.end(), base_v1.begin(), base_v1.end());
    }

    // generate level
    std::vector<float> level_base(base_v1);
    std::vector<float> level_hvs;
    for (int q = 0; q <= Q; ++q) {
        int flip = (int) (q/float(Q) * dim) / 2;
        std::vector<float> level_hv(level_base);
        // + flip will transform (flip) number of elements
        std::transform(level_hv.begin(), level_hv.begin() + flip, level_hv.begin(), bind2nd(std::multiplies<float>(), -1)); 
        level_hvs.insert(level_hvs.end(), level_hv.begin(), level_hv.end());
    }

    // generate id
    std::shuffle(level_base.begin(), level_base.end(), std::default_random_engine(seed));  // use this as id_base
    std::vector<float> id_hvs(level_base);  // f=0
    for (int f = 1; f < nFeatures_train; ++f) {
        std::rotate(level_base.begin(), level_base.begin() + 1, level_base.end());
        id_hvs.insert(id_hvs.end(), level_base.begin(), level_base.end());
    }

    //////////////////////////////////////////////////////////////////////////
    // GPU LOAD
    // int nThreads = N_THREADS;
    // int nBlocks = int(ceil(float(features) / float(nThreads)));

    cudaEvent_t dataload, start, stop1, stop2, stop3, stop4;
    cudaError_t err = cudaSuccess;

    cudaEventCreate(&dataload);
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    cudaEventCreate(&stop3);
    cudaEventCreate(&stop4);

    float* d_bases = NULL;
    int* d_y_train = NULL;
    int* d_y_test = NULL;
    float* d_x_train = NULL;
    float* d_x_test = NULL;
    float* d_hvs_train = NULL;
    float* d_hvs_test = NULL;
#ifdef USE_COS_SIMILARITY
    float* d_train_norm = NULL;
    float* d_test_norm = NULL;
    float* d_weights_norm = NULL;
#endif

    cudaEventRecord(dataload);

    HANDLE_ERROR(cudaMalloc((void **)&d_bases, base_size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_y_train, y_train.size() * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&d_y_test, y_test.size() * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&d_x_train, x_train_flat.size() * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_x_test, x_test_flat.size() * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_hvs_train, 4 * train_encode_size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_hvs_test, 4 * test_encode_size * sizeof(float)));

#ifdef USE_COS_SIMILARITY
    HANDLE_ERROR(cudaMalloc((void **)&d_train_norm, train_set_num * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_test_norm, test_set_num * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_weights_norm, nClasses_train * sizeof(float)));
#endif

    HANDLE_ERROR(cudaMemcpy(d_bases, bases.data(), base_size * sizeof(float), cudaMemcpyHostToDevice));  //convert vector to array
    HANDLE_ERROR(cudaMemcpy(d_y_train, y_train.data(), y_train.size() * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_y_test, y_test.data(), y_test.size() * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_x_train, x_train_flat.data(), x_train_flat.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_x_test, x_test_flat.data(), x_test_flat.size() * sizeof(float), cudaMemcpyHostToDevice));

    //id level
    float* d_level_hvs = NULL;
    float* d_id_hvs = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_level_hvs, level_hvs.size() * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_id_hvs, id_hvs.size() * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_level_hvs, level_hvs.data(), level_hvs.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_id_hvs, id_hvs.data(), id_hvs.size() * sizeof(float), cudaMemcpyHostToDevice));


    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1;
    const float beta = 0;

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    printf("Starting Encoding Stage...\n");

#ifdef USE_DOT_ENCODING
    // Encode stage: Linear
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
               dim, train_set_num, nFeatures_train, 
               &alpha, d_bases, dim, 
               d_x_train, nFeatures_train, &beta, 
               d_hvs_train, dim);
    cudaThreadSynchronize();
#endif
#ifdef USE_LVID_ENCODING
    dim3 encodeblocksTrain((dim + N_THREADS - 1) / N_THREADS, train_set_num);
    dim3 encodeTPB(N_THREADS, 1, 1);

    int level_stride = dim * 4;
    int id_stride = dim * 4;
    int fm_stride = nFeatures_train * 4;

    encodeLevelId<<<encodeblocksTrain, encodeTPB>>>(d_level_hvs, d_id_hvs, d_x_train, d_hvs_train, level_stride, 
                                                id_stride, fm_stride, train_set_num, nFeatures_train, Q, dim);
#endif

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    // Training stage
    // TODO: Add validation
    // make_guess and create guess table
    printf("Training stage...\n");

    float* d_guess_vec = NULL;
    float* d_weights = NULL;

    int guess_hit_training = 0;

    HANDLE_ERROR(cudaMalloc((void **)&d_guess_vec, nClasses_train * sizeof(float)));  // show prob. for class for each case
    HANDLE_ERROR(cudaMalloc((void **)&d_weights, nClasses_train * dim * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_weights, 0, nClasses_train * dim * sizeof(float)));

#ifdef USE_COS_SIMILARITY
    HANDLE_ERROR(cudaMemset(d_weights_norm, 0, nClasses_train * sizeof(float)));
    normMatRow<<<(train_set_num + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_train_norm, d_hvs_train, train_set_num, dim);
#endif

#pragma unroll
    for (int iter = 0; iter < iter_num; ++iter) {  // Retraining
#pragma unroll
        for (int ii = 0; ii < train_set_num; ++ii) {
            // TODO: Implement batch
            cublasSgemv(handle, CUBLAS_OP_T, 
                        dim, nClasses_train, 
                        &alpha, d_weights, dim, 
                        d_hvs_train + ii * dim, 1, &beta, 
                        d_guess_vec, 1);   // np.dot
            cudaThreadSynchronize();

#ifdef USE_COS_SIMILARITY
            cosineSimilarityVec<<<(nClasses_train + N_THREADS - 1)/N_THREADS, N_THREADS>>>(d_guess_vec, d_weights_norm, nClasses_train, d_train_norm, ii);
            // cosineSimilarityVec_woreuse<<<(nClasses_train + N_THREADS - 1)/N_THREADS, N_THREADS>>>(d_guess_vec, d_weights, dim, nClasses_train, d_train_norm, ii);
#endif

            int guess = -1;
            findargmax<<<1, 1>>>(d_guess_vec, nClasses_train);
            cudaMemcpyFromSymbol(&guess, d_guess, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            // update weight matrix
            if (guess != y_train[ii]) {  // TODO: is this safe?
                cublasSaxpy(handle, dim, &neg_learning_rate, d_hvs_train + ii * dim, 1, d_weights + guess * dim, 1);
                cublasSaxpy(handle, dim, &learning_rate, d_hvs_train + ii * dim, 1, d_weights + y_train[ii] * dim, 1);
                cudaThreadSynchronize();  //TODO: Is this safe?
#ifdef USE_COS_SIMILARITY
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
                cublasSnrm2(handle, dim, d_hvs_train + ii * dim, 1, d_weights_norm + y_train[ii]);
                cublasSnrm2(handle, dim, d_hvs_train + ii * dim, 1, d_weights_norm + guess);
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
#endif
            }
            else
                guess_hit_training++;
        }
    }
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    // Testing stage: d_hvs_test vs d_weights (classes * dim)
    printf("Starting Test Stage..\n");

    #ifdef USE_DOT_ENCODING
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
               dim, test_set_num, nFeatures_test, 
               &alpha, d_bases, dim, 
               d_x_test, nFeatures_test, &beta, 
               d_hvs_test, dim);
    cudaThreadSynchronize();
#endif
#ifdef USE_LVID_ENCODING
    dim3 encodeblocksTest((dim + N_THREADS - 1) / N_THREADS, test_set_num);
    encodeLevelId<<<encodeblocksTest, encodeTPB>>>(d_level_hvs, d_id_hvs, d_x_test, d_hvs_test, level_stride, 
                                                id_stride, fm_stride, test_set_num, nFeatures_test, Q, dim);
#endif

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);

    float* d_guess_vec_test = NULL;
    bool* d_scoreboard = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_guess_vec_test, test_set_num * nClasses_test * sizeof(float)));  // show prob. for class for each case
    HANDLE_ERROR(cudaMalloc((void **)&d_scoreboard, test_set_num * sizeof(bool)));

#ifdef USE_COS_SIMILARITY
    normMatRow<<<(test_set_num + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_test_norm, d_hvs_test, test_set_num, dim);
#endif

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        nClasses_test, test_set_num, dim, 
        &alpha, d_weights, dim, 
        d_hvs_test, dim, &beta, 
        d_guess_vec_test, nClasses_test);
    cudaThreadSynchronize();

#ifdef USE_COS_SIMILARITY
    guessVecGenCompareCosine<<<(test_set_num + N_THREADS - 1) / N_THREADS, N_THREADS>>>
        (d_scoreboard, d_y_test, d_weights_norm, d_test_norm, d_guess_vec_test, test_set_num, nClasses_test);
#endif
#ifdef USE_DOT_SIMILARITY
    guessVecGenCompareDot<<<(test_set_num + N_THREADS - 1) / N_THREADS, N_THREADS>>>
        (d_scoreboard, d_y_test, d_guess_vec_test, test_set_num, nClasses_test);
#endif
    cudaEventRecord(stop4);
    cudaEventSynchronize(stop4);

    std::cout << "Train Acc: " << (float) guess_hit_training/(train_set_num * iter_num) << std::endl;
    
    int guess_hit_testing = 0;
    bool* scoreboard = (bool*)malloc(test_set_num * sizeof(bool));
    HANDLE_ERROR(cudaMemcpy(scoreboard, d_scoreboard, test_set_num * sizeof(bool), cudaMemcpyDeviceToHost));
    for (int jj = 0; jj < test_set_num; ++jj) {
        if (scoreboard[jj] == 1)
            guess_hit_testing++;
    }
    std::cout << "Test Acc: " << (float) guess_hit_testing/test_set_num << std::endl;

    cublasDestroy(handle);
    HANDLE_ERROR(cudaFree(d_bases));
    HANDLE_ERROR(cudaFree(d_y_train));
    HANDLE_ERROR(cudaFree(d_y_test));
    HANDLE_ERROR(cudaFree(d_x_train));
    HANDLE_ERROR(cudaFree(d_x_test));
    HANDLE_ERROR(cudaFree(d_hvs_train));
    HANDLE_ERROR(cudaFree(d_hvs_test));
    HANDLE_ERROR(cudaFree(d_guess_vec));
    HANDLE_ERROR(cudaFree(d_weights));

#ifdef USE_COS_SIMILARITY
    HANDLE_ERROR(cudaFree(d_train_norm));
    HANDLE_ERROR(cudaFree(d_test_norm));
    HANDLE_ERROR(cudaFree(d_weights_norm));
#endif

    HANDLE_ERROR(cudaFree(d_guess_vec_test));
    HANDLE_ERROR(cudaFree(d_scoreboard));

    HANDLE_ERROR(cudaFree(d_level_hvs));
    HANDLE_ERROR(cudaFree(d_id_hvs));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, dataload, start);
    printf("GPU Execution time (Data Loading): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, start, stop1);
    printf("GPU Execution time (Encoding): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, stop1, stop2);
    printf("GPU Execution time (Training): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, stop2, stop3);
    printf("GPU Execution time (QueryEnc): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, stop3, stop4);
    printf("GPU Execution time (Test): %f\n", milliseconds);

    cudaEventElapsedTime(&milliseconds, dataload, stop4);
    printf("GPU Execution time (End-to-end): %f\n", milliseconds);
    
    cudaEventDestroy(dataload);
    cudaEventDestroy(start);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);
    cudaEventDestroy(stop3);
    cudaEventDestroy(stop4);
    
    free(scoreboard);
    return 0;
}
