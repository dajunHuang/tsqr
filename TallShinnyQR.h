#pragma once

#include <cusolverDn.h>

#include <cmath>
#include <iostream>

#include "kernelQR.h"
#include "utils.h"

template <typename T>
__global__ void tsgemm(int m, int n, T *A, const int lda, T *B, const int ldb,
                       T *C, const int ldc) {
    const int block_num = gridDim.x;
    const int block_dim_x = BLOCK_DIM_X, block_dim_y = BLOCK_DIM_Y;
    const int block_idx = blockIdx.x;
    const int thread_idx_x = threadIdx.x, thread_idx_y = threadIdx.y;
    const int num_row =
        (m + block_num * block_dim_x - 1) / block_num * block_dim_x;
    const int num_col =
        (n + block_dim_y - 1) / block_dim_y;
    
    T c_per_thread[NUM_Q_ROW * NUM_Q_COL];

    for (int row_repeat_idx = 0; row_repeat_idx < num_row; ++row_repeat_idx) {
        int row_idx = row_repeat_idx * block_num * block_dim_x +
                      block_idx * block_dim_x + thread_idx_x;
        if (row_idx >= m) break;
        for (int col_repeat_idx = 0; col_repeat_idx < num_col;
             ++col_repeat_idx) {
            int col_idx = col_repeat_idx * block_dim_y + thread_idx_y;
            if (col_idx >= n) break;
            T sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += A[row_idx + k * lda] * B[k + col_idx * ldb];
            }
            c_per_thread[row_repeat_idx + col_repeat_idx * NUM_Q_ROW] = sum;
        }
    }

    for (int row_repeat_idx = 0; row_repeat_idx < num_row; ++row_repeat_idx) {
        int row_idx = row_repeat_idx * block_num * block_dim_x +
                      block_idx * block_dim_x + thread_idx_x;
        if (row_idx >= m) break;
        for (int col_repeat_idx = 0; col_repeat_idx < num_col;
             ++col_repeat_idx) {
            int col_idx = col_repeat_idx * block_dim_y + thread_idx_y;
            if (col_idx >= n) break;
            C[row_idx + col_idx * ldc] = c_per_thread[row_repeat_idx + col_repeat_idx * NUM_Q_ROW];
        }
    }
}
template __global__ void tsgemm<double>(int m, int n, double *A, const int lda,
                                        double *B, const int ldb, double *C,
                                        const int ldc);
template __global__ void tsgemm<float>(int m, int n, float *A, const int lda,
                                       float *B, const int ldb, float *C,
                                       const int ldc);

template <typename T>
void tsqr(int m, int n, T *A, int lda, T *R, int ldr, T *d_work, int ldwork) {
    int max_grid_size = NUM_SM * BLOCK_SIZE;
    T *d_work1 = d_work, *d_work2 = d_work + max_grid_size;

    dim3 block_dim(
        BLOCK_DIM_X,
        BLOCK_DIM_Y);  // if change block_dim, also change acc_per_thread
                       // and q_per_thread mannually in kernelQR.h
    if (m > (max_grid_size / n) * max_grid_size) {
        printf("not supported size\n");
        return;
    }
    int grid_num = (m + max_grid_size - 1) / max_grid_size;

    if (grid_num > 1) {
        assert((m % max_grid_size) % n == 0);
        assert(BLOCK_SIZE % n == 0);

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

            tsqr_kernel<T><<<block_num, block_dim, share_memory_size>>>(
                grid_size, n, &A[i * max_grid_size], lda, &d_work1[i * n],
                ldwork, d_work2, ldwork);
        }

        int block_num = (grid_num * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        tsqr_kernel<T><<<block_num, block_dim, share_memory_size>>>(
            grid_num * n, n, d_work1, ldwork, R, ldr, d_work2, ldwork);

        block_num = NUM_SM;
        for (int i = 0; i < grid_num; ++i) {
            int grid_size = min(m - i * max_grid_size, max_grid_size);
            tsgemm<<<block_num, block_dim>>>(
                grid_size, n, &A[i * max_grid_size], lda, &d_work1[i * n],
                ldwork, &A[i * max_grid_size], lda);
        }
    } else {
        assert((m % BLOCK_SIZE) % n == 0);
        assert(BLOCK_SIZE % n == 0);

        int reduction_time =
            ceil((log(m) - log(n)) / (log(BLOCK_SIZE) - log(n)));
        // printf("size %d, reduction_time: %d\n", m, reduction_time);
        int share_memory_size = reduction_time * BLOCK_SIZE * n * sizeof(T);

        CUDA_CHECK(cudaFuncSetAttribute(
            tsqr_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            share_memory_size));

        int block_num = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        tsqr_kernel<T><<<block_num, block_dim, share_memory_size>>>(
            m, n, A, lda, R, ldr, d_work2, ldwork);
    }
}
template void tsqr<double>(int m, int n, double *A, int lda, double *R, int ldr,
                           double *d_work, int ldwork);
template void tsqr<float>(int m, int n, float *A, int lda, float *R, int ldr,
                          float *d_work, int ldwork);
