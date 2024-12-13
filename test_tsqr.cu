#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "TallShinnyQR.h"
#include "TallShinnyQR_ori.h"

#define NUM_WARPUP 20
#define NUM_REPEAT 50

template <typename T>
void test_tsqr(int block_size, int m, int n) {
    // cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int lda = m;
    const int ldr = n;
    double one = 1, zero = 0;

    std::vector<T> A(m * n, 0);
    std::vector<T> A_from_gpu(m * n, 0);
    std::vector<T> R_from_gpu(n * n, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<T> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    std::generate(A.begin(), A.end(), rand);

    T *d_A = nullptr;
    T *d_R = nullptr;
    T *d_work1 = nullptr;
    T *d_work2 = nullptr;
    T *d_work_ori = nullptr;

    /* step 1: create cusolver handle, bind a stream */
    // CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(T) * n * n));
    // CUDA_CHECK(
    //     cudaMalloc(reinterpret_cast<void **>(&d_work_ori), sizeof(T) * m *
    //     m));

    const int ldwork1 = 108 * block_size;
    const int ldwork2 = m;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work1),
                          sizeof(T) * ldwork1 * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work2), sizeof(T) * m * n));
    // # origianl
    // CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
    //                       cudaMemcpyHostToDevice));
    // printf("\ntsqr_ori\n");
    // tsqr_ori<T, 128, 32>(cublasH, m, n, d_A, lda, d_R, ldr,
    //                                d_work_ori);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK_LAST_ERROR();
    // printf("R\n");
    // printDeviceMatrixV2(d_R, ldr, 32, 32);
    // printf("Q\n");
    // printDeviceMatrixV2(d_A, lda, m < 169 ? m : 169, 32);

    // # new
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
                          cudaMemcpyHostToDevice));
    // printf("A\n");
    // printDeviceMatrixV2(d_A, lda, m < 169 ? m : 169, 32);
    // printf("tsqr\n");
    tsqr<T>(cublasH, block_size, m, n, d_A, lda, d_R, ldr, d_work1,
                      ldwork1, d_work2, ldwork2);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST_ERROR();
    // printf("R\n");
    // printDeviceMatrixV2(d_R, ldr, 32, 32);
    // printf("Q\n");
    // printDeviceMatrixV2(d_A, lda, m < 32 ? m : 32, n < 32 ? n : 32);

    // T *d_Q = nullptr;
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q), sizeof(T) * n *
    // n)); cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &one, d_A,
    //             lda, d_A, lda, &zero, d_Q, n);
    // printDeviceMatrixV2(d_Q, n, n, n);

    cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one, d_A, lda,
                d_R, ldr, &zero, d_A, lda);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(A_from_gpu.data(), d_A,
                               sizeof(T) * A_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!all_close(A_from_gpu.data(), A.data(), m, n, lda, 1.0e-4, 1.0e-5)) {
        std::cout << "Error: tsqr" << std::endl;
        exit(-1);
    }

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        tsqr<T>(cublasH, block_size, m, n, d_A, lda, d_R, ldr,
                          d_work1, ldwork1, d_work2, ldwork2);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start, stream));

        tsqr<T>(cublasH, block_size, m, n, d_A, lda, d_R, ldr,
                          d_work1, ldwork1, d_work2, ldwork2);

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaMemcpyAsync(A_from_gpu.data(), d_A,
                               sizeof(T) * A_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(R_from_gpu.data(), d_R,
                               sizeof(T) * R_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "tsqr Latency: " << time << " ms" << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_work1));
    CUDA_CHECK(cudaFree(d_work2));
    CUDA_CHECK(cudaFree(d_work_ori));

    // CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());
}

// template void test_tsqr<float>(int m, int n);
template void test_tsqr<double>(int block_size, int m, int n);

int main(int argc, char *argv[]) {
    int m = 13824, n = 32;
    int block_size = 128;
    int dataType = 2;

    // print_device_info();

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        dataType = atoi(argv[3]);
    }

    if (0 == dataType) {
        // test_tsqr<half>(m, n);
    } else if (1 == dataType) {
        // test_tsqr<float>(m, n);
    } else if (2 == dataType) {
        test_tsqr<double>(block_size, m, n);
    }

    return 0;
}
