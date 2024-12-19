#pragma once

#include <cusolverDn.h>

#include <cmath>
#include <iostream>

#include "kernelQR.h"
#include "tsgemm.h"
#include "utils.h"

template <typename T>
void tsqr(int m, int n, T *A, int lda, T *R, int ldr, T *d_work, int ldwork) {
    int max_grid_size = NUM_SM * BLOCK_SIZE;
    T *d_work1 = d_work, *d_work2 = d_work + max_grid_size;

    assert(m >= n);
    assert(BLOCK_SIZE % n == 0);

    dim3 block_dim_gemm{32, 32};
    dim3 block_dim_tsqr(
        BLOCK_DIM_X,
        BLOCK_DIM_Y);  // if change block_dim_tsqr, also change acc_per_thread
                       // and q_per_thread mannually in kernelQR.h
    if (m > (max_grid_size / n) * max_grid_size) {
        printf("not supported size\n");
        return;
    }
    // printf("max_grid_size: %d, max_supported_size: %d\n", max_grid_size,
    //        (max_grid_size / n) * max_grid_size);
    int grid_num = (m + max_grid_size - 1) / max_grid_size;

    if (grid_num > 1) {
        assert((m % max_grid_size) % n == 0);

        int reduction_time =
            ceil((log(max_grid_size) - log(n)) / (log(BLOCK_SIZE) - log(n)));
        // printf("size %d, reduction_time: %d\n", m, reduction_time);
        int share_memory_size = reduction_time * BLOCK_SIZE * n * sizeof(T);
        CUDA_CHECK(cudaFuncSetAttribute(
            tsqr_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            share_memory_size));

        for (int i = 0; i < grid_num; ++i) {
            int grid_size = min(m - i * max_grid_size, max_grid_size);
            int block_num = (grid_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            tsqr_kernel<T><<<block_num, block_dim_tsqr, share_memory_size>>>(
                grid_size, n, &A[i * max_grid_size], lda, &d_work1[i * n],
                ldwork, d_work2, ldwork);
        }

        int block_num =
            (grid_num * n + block_dim_gemm.x - 1) / block_dim_gemm.x;
        tsqr_kernel<T><<<block_num, block_dim_tsqr, share_memory_size>>>(
            grid_num * n, n, d_work1, ldwork, R, ldr, d_work2, ldwork);

        block_num = NUM_SM;
        for (int i = 0; i < grid_num; ++i) {
            int grid_size = min(m - i * max_grid_size, max_grid_size);
            tsgemm<T><<<block_num, block_dim_gemm>>>(
                grid_size, n, n, 1.0, 0, &A[i * max_grid_size], lda,
                &d_work1[i * n], ldwork, 0.0, &A[i * max_grid_size], lda);
        }
    } else {
        assert((m % BLOCK_SIZE) % n == 0);

        int reduction_time = 0;
        if (m == n) {
            reduction_time = 1;
        } else {
            reduction_time =
                ceil((log(m) - log(n)) / (log(BLOCK_SIZE) - log(n)));
        }
        // printf("size %d, reduction_time: %d\n", m, reduction_time);
        int share_memory_size = reduction_time * BLOCK_SIZE * n * sizeof(T);

        CUDA_CHECK(cudaFuncSetAttribute(
            tsqr_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            share_memory_size));

        int block_num = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        tsqr_kernel<T><<<block_num, block_dim_tsqr, share_memory_size>>>(
            m, n, A, lda, R, ldr, d_work2, ldwork);
    }
}
template void tsqr<double>(int m, int n, double *A, int lda, double *R, int ldr,
                           double *d_work, int ldwork);
template void tsqr<float>(int m, int n, float *A, int lda, float *R, int ldr,
                          float *d_work, int ldwork);
