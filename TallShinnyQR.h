#pragma once

#include <cusolverDn.h>

#include <cmath>
#include <iostream>

#include "kernelQR.h"
#include "utils.h"

template <typename T>
__global__ void tsgemm_tsqr(int m_total, int m_each_max, int n, int k, T alpha,
                            T *A, const int lda, T *B, const int ldb, T beta,
                            T *C, const int ldc) {
    const int grid_dim_x = gridDim.x, grid_dim_y = gridDim.y;
    const int block_dim_x = blockDim.x, block_dim_y = blockDim.y;
    const int block_idx_x = blockIdx.x, block_idx_y = blockIdx.y;
    const int thread_idx_x = threadIdx.x, thread_idx_y = threadIdx.y;

    const int num_col =
        (n + grid_dim_y * block_dim_y - 1) / grid_dim_y * block_dim_y;

    T c_per_thread[NUM_Q_ROW * NUM_Q_COL];

    int m = 0;
    const int grid_num = (m_total + m_each_max - 1) / m_each_max;

    for (int grid_idx = 0; grid_idx < grid_num; ++grid_idx) {
        m = min(m_total - grid_idx * m_each_max, m_each_max);
        int num_row =
            (m + grid_dim_x * block_dim_x - 1) / grid_dim_x * block_dim_x;

        for (int row_repeat_idx = 0; row_repeat_idx < num_row;
             ++row_repeat_idx) {
            int row_idx = row_repeat_idx * grid_dim_x * block_dim_x +
                          block_idx_x * block_dim_x + thread_idx_x;
            if (row_idx >= m) break;
            for (int col_repeat_idx = 0; col_repeat_idx < num_col;
                 ++col_repeat_idx) {
                int col_idx = col_repeat_idx * grid_dim_y * block_dim_y +
                              block_idx_y * block_dim_y + thread_idx_y;
                if (col_idx >= n) break;
                T sum = 0;
                for (int i = 0; i < k; ++i) {
                    sum += A[row_idx + i * lda] * B[i + col_idx * ldb];
                }
                c_per_thread[row_repeat_idx + col_repeat_idx * NUM_Q_ROW] = sum;
            }
        }

        __syncthreads();

        for (int row_repeat_idx = 0; row_repeat_idx < num_row;
             ++row_repeat_idx) {
            int row_idx = row_repeat_idx * grid_dim_x * block_dim_x +
                          block_idx_x * block_dim_x + thread_idx_x;
            if (row_idx >= m) break;
            for (int col_repeat_idx = 0; col_repeat_idx < num_col;
                 ++col_repeat_idx) {
                int col_idx = col_repeat_idx * grid_dim_y * block_dim_y +
                              block_idx_y * block_dim_y + thread_idx_y;
                if (col_idx >= n) break;
                C[row_idx + col_idx * ldc] =
                    alpha * c_per_thread[row_repeat_idx +
                                         col_repeat_idx * NUM_Q_ROW] +
                    beta * C[row_idx + col_idx * ldc];
            }
        }

        A += m;
        B += n;
        C += m;
    }
}
template __global__ void tsgemm_tsqr<double>(int m_total, int m_each_max, int n,
                                             int k, double alpha, double *A,
                                             const int lda, double *B,
                                             const int ldb, double beta,
                                             double *C, const int ldc);
template __global__ void tsgemm_tsqr<float>(int m_total, int m_each_max, int n,
                                            int k, float alpha, float *A,
                                            const int lda, float *B,
                                            const int ldb, float beta, float *C,
                                            const int ldc);

template <typename T>
void tsqr(int m, int n, T *A, int lda, T *R, int ldr, T *d_work, int ldwork) {
    int block_num = NUM_SM;
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

        tsqr_kernel<T><<<block_num, block_dim_tsqr, share_memory_size>>>(
            m, max_grid_size, n, A, lda, d_work1, ldwork, d_work2, ldwork);

        tsqr_kernel<T><<<block_num, block_dim_tsqr, share_memory_size>>>(
            grid_num * n, grid_num * n, n, d_work1, ldwork, R, ldr, d_work2,
            ldwork);

        tsgemm_tsqr<T><<<block_num, block_dim_gemm>>>(
            m, max_grid_size, n, n, 1.0, A, lda, d_work1, ldwork, 0.0, A, lda);
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
            m, m, n, A, lda, R, ldr, d_work2, ldwork);
    }
}
template void tsqr<double>(int m, int n, double *A, int lda, double *R, int ldr,
                           double *d_work, int ldwork);
template void tsqr<float>(int m, int n, float *A, int lda, float *R, int ldr,
                          float *d_work, int ldwork);
