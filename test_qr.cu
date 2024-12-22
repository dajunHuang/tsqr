#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "block_QR.h"

#define NUM_WARPUP 2
#define NUM_REPEAT 5

template <typename T>
void test_qr(int m, int n) {
    cudaStream_t stream = NULL;

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
    T *d_A = nullptr;
    T *d_R = nullptr;
    T *d_work = nullptr;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(T) * n * n));

    const int ldwork = 2 * NUM_SM * BLOCK_SIZE;

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * ldwork * n));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
                          cudaMemcpyHostToDevice));
    // printf("A\n");
    // print_device_matrix(d_A, lda, m < 169 ? m : 169, 32);
    // printf("qr\n");
    qr<T>(m, n, d_A, lda, d_R, ldr, d_work, ldwork);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST_ERROR();
    // printf("R\n");
    // print_device_matrix(d_R, ldr, 32, 32);
    // printf("Q\n");
    // print_device_matrix(d_A, lda, m < 32 ? m : 32, n < 32 ? n : 32);

    check_QR_accuracy<T>(m, n, d_A, lda, d_R, ldr, A);

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        qr<T>(m, n, d_A, lda, d_R, ldr, d_work, ldwork);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start, stream));

        qr<T>(m, n, d_A, lda, d_R, ldr, d_work, ldwork);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    printf("qr Latency: %f ms\n", time);

    CUDA_CHECK(cudaMemcpyAsync(A_from_gpu.data(), d_A,
                               sizeof(T) * A_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(R_from_gpu.data(), d_R,
                               sizeof(T) * R_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_work));

    CUDA_CHECK(cudaDeviceReset());
}
template void test_qr<double>(int m, int n);
template void test_qr<float>(int m, int n);

int main(int argc, char *argv[]) {
    int m = 13824, n = 32;
    int dataType = 2;

    // print_device_info();

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        dataType = atoi(argv[3]);
    }

    if (1 == dataType) {
        test_qr<float>(m, n);
    } else if (2 == dataType) {
        test_qr<double>(m, n);
    }

    return 0;
}
