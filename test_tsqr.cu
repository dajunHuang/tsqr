#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "TallShinnyQR.h"

#define NUM_WARPUP 20
#define NUM_REPEAT 50

template <typename T>
void test_tsqr(int m, int n) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    double one = 1, zero = 0, minus_one = -1;

    std::vector<T> A(m * n, 0);
    std::vector<T> A_from_gpu(m * n, 0);
    std::vector<T> R_from_gpu(n * n, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<T> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    std::generate(A.begin(), A.end(), rand);

    const int lda = m;
    const int ldr = n;
    const int ldqtq = n;
    const int ldqr = m;
    T *d_A = nullptr;
    T *d_R = nullptr;
    T *d_QTQ = nullptr;
    T *d_QR = nullptr;
    T *d_work = nullptr;

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(T) * n * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_QTQ), sizeof(T) * n * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_QR), sizeof(T) * m * n));

    const int ldwork = m + NUM_SM * BLOCK_SIZE; // m + max_grid_size

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work),
                          sizeof(T) * ldwork * n));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
                          cudaMemcpyHostToDevice));
    // printf("A\n");
    // print_device_matrix(d_A, lda, m < 169 ? m : 169, 32);
    // printf("tsqr\n");
    tsqr<T>(m, n, d_A, lda, d_R, ldr, d_work, ldwork);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST_ERROR();
    // printf("R\n");
    // print_device_matrix(d_R, ldr, 32, 32);
    // printf("Q\n");
    // print_device_matrix(d_A, lda, m < 32 ? m : 32, n < 32 ? n : 32);

    // QR = Q * R
    cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one, d_A, lda, d_R,
                ldr, &zero, d_QR, ldqr);

    // move QR to host memory
    CUDA_CHECK(cudaMemcpyAsync(A_from_gpu.data(), d_QR,
                               sizeof(T) * A_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // compare QR with original A
    if (!all_close(A_from_gpu.data(), A.data(), m, n, lda, 1.0e-4, 1.0e-5)) {
        std::cout << "Error: tsqr" << std::endl;
        exit(-1);
    }

    // d_QTQ = I - Q^T * Q
    init_identity_matrix<<<1, 1>>>(d_QTQ, ldqtq, n, n);
    cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &minus_one, d_A,
                lda, d_A, lda, &one, d_QTQ, ldqtq);
    T QTQ_2_norm = get_matrix_2_norm(cusolverH, n, n, d_QTQ, ldqtq);

    // d_QR = A
    CUDA_CHECK(cudaMemcpy(d_QR, A.data(), sizeof(T) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    T A_2_norm = get_matrix_2_norm(cusolverH, m, n, d_QR, ldqr);

    // d_QR = A - QR
    CUDA_CHECK(cudaMemcpy(d_QR, A.data(), sizeof(T) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &minus_one, d_A,
                lda, d_R, ldr, &one, d_QR, ldqr);
    T QR_2_norm = get_matrix_2_norm(cusolverH, m, n, d_QR, ldqr);

    CUDA_CHECK(cudaFree(d_QTQ));
    CUDA_CHECK(cudaFree(d_QR));

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        tsqr<T>(m, n, d_A, lda, d_R, ldr, d_work, ldwork);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start, stream));

        tsqr<T>(m, n, d_A, lda, d_R, ldr, d_work, ldwork);

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

    printf("|A-QR|/|A| = %.17f, |I-Q^TQ| = %.17f\ntsqr Latency: %f ms\n",
           QR_2_norm / A_2_norm, QTQ_2_norm, time);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_work));
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());
}
template void test_tsqr<double>(int m, int n);

int main(int argc, char *argv[]) {
    int m = 13824, n = 32;
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
        test_tsqr<double>(m, n);
    }

    return 0;
}
