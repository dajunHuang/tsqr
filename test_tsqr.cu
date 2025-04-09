#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "TallShinnyQR.h"
#include "test_utils.h"

#define NUM_WARPUP 2
#define NUM_REPEAT 5

template <typename T>
void test_tsqr(long m, long n) {
    // cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const long lda = m;
    const long ldr = n;

    T *d_A_ori = nullptr;
    T *d_A = nullptr;
    T *d_R = nullptr;
    T *d_work = nullptr;

    /* step 1: create cusolver handle, bind a stream */
    // CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_ori), sizeof(T) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(T) * ldr * n));

    const int ldwork = 8192;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * ldwork * n * 16));

    generateUniformMatrix(d_A_ori, lda, n);
    CUDA_CHECK(cudaMemcpy(d_A, d_A_ori, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));

    tsqr<T>(cublasH, m, n, d_A, lda, d_R, ldr, d_work, ldwork);
    CUDA_CHECK(cudaDeviceSynchronize());

    check_QR_accuracy<T>(m, n, d_A, lda, d_R, ldr, d_A_ori);

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        CUDA_CHECK(cudaMemcpy(d_A, d_A_ori, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        tsqr<T>(cublasH, m, n, d_A, lda, d_R, ldr, d_work, ldwork);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaMemcpy(d_A, d_A_ori, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start, stream));

        tsqr<T>(cublasH, m, n, d_A, lda, d_R, ldr, d_work, ldwork);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    std::cout << "m: " << m << ", n: " << n << ", hou_tsqr_panel Latency: " << time << " ms, "
              << (float(2) * m * n * n - float(2) / 3 * n * n * n) / time / 1e9 << " TFLOPS"
              << std::endl;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_work));

    // CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());
}

template void test_tsqr<float>(long m, long n);
template void test_tsqr<double>(long m, long n);

int main(int argc, char *argv[]) {
    long m = 16384, n = 32, dataType = 2;

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        dataType = atoi(argv[3]);
    }

    if (0 == dataType) {
        // test_hou_tsqr_panel<half>(m, n);
    } else if (1 == dataType) {
        test_tsqr<float>(m, n);
    } else if (2 == dataType) {
        test_tsqr<double>(m, n);
    }

    return 0;
}
