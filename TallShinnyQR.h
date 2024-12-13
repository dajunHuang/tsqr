#pragma once

#include <cusolverDn.h>

#include <cmath>
#include <iostream>

#include "kernelQR.h"
#include "myBase.h"

__global__ void tsgemm(int m, int n, double *A, const int lda, double *B,
                           const int ldb, double *C, const int ldc) {
    const int grid_dim_x = gridDim.x, grid_dim_y = gridDim.y;
    const int block_dim_x = blockDim.x, block_dim_y = blockDim.y;
    const int block_idx_x = blockIdx.x, block_idx_y = blockIdx.y;
    const int thread_idx_x = threadIdx.x, thread_idx_y = threadIdx.y;

    const int num_row = (m + grid_dim_x * block_dim_x - 1) / grid_dim_x * block_dim_x;
    const int num_col = (n + grid_dim_y * block_dim_y - 1) / grid_dim_y * block_dim_y;
    for (int row_repeat_idx = 0; row_repeat_idx < num_row; ++row_repeat_idx) {
        int row_idx = row_repeat_idx * grid_dim_x * block_dim_x + block_idx_x * block_dim_x + thread_idx_x;
        if (row_idx >= m) break;
        for (int col_repeat_idx = 0; col_repeat_idx < num_col;
             ++col_repeat_idx) {
            int col_idx = col_repeat_idx * grid_dim_y * block_dim_y + block_idx_y * block_dim_y + thread_idx_y;
            if (col_idx >= n) break;
            double sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += A[row_idx + k * lda] * B[k + col_idx * ldb];
            }
            C[row_idx + col_idx * ldc] = sum;
        }
    }
}

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T>
void tsqr(cublasHandle_t cublas_handle, int block_size, int m, int n,
                    T *A, int lda, T *R, int ldr, T *d_work1, int ldwork1,
                    T *d_work2, int ldwork2) {
    dim3 block_dim(32, 32);  // if change block_dim, also change acc_per_thread
                             // and q_per_thread mannually in kernelQR.h
    int max_grid_size = 108 * block_size;
    if (m > (max_grid_size / n) * max_grid_size) {
        printf("not supported size\n");
        return;
    }
    int grid_num = (m + max_grid_size - 1) / max_grid_size;

    if (grid_num > 1) {
        assert((m % max_grid_size) % n == 0);
        assert(block_size % n == 0);

        int reduction_time =
            ceil((log(max_grid_size) - log(n)) / (log(block_size) - log(n)));
        // printf("size %d, reduction_time: %d\n", m, reduction_time);
        int share_memory_size = reduction_time * block_size * n * sizeof(T);
        CUDA_CHECK(cudaFuncSetAttribute(
            householder_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            share_memory_size));

        for (int i = 0; i < grid_num; ++i) {
            int grid_size = min(m - i * max_grid_size, max_grid_size);

            int block_num = (grid_size + block_size - 1) / block_size;
            T *grid_A = A + i * max_grid_size, *grid_R = d_work1 + i * n;
            int ldgrid_A = lda, ldgrid_R = ldwork1;

            householder_kernel<T><<<block_num, block_dim, share_memory_size>>>(
                block_size, grid_size, n, grid_A, ldgrid_A, grid_R, ldgrid_R,
                d_work2, ldwork2);
        }
        int block_num = (grid_num * n + block_size - 1) / block_size;
        householder_kernel<T><<<block_num, block_dim, share_memory_size>>>(
            block_size, grid_num * n, n, d_work1, ldwork1, R, ldr, d_work2,
            ldwork2);

        int grid_dim = 108;
        dim3 block_dim = {32, 32};
        for(int i = 0; i < grid_num - 1; ++i) {
            int grid_size = min(m - i * max_grid_size, max_grid_size);
            tsgemm<<<grid_dim, block_dim>>>(grid_size, n, A + i * max_grid_size, lda,
                                            d_work1 + i * n, ldwork1, A + i * max_grid_size, lda);
        }

        int last_grid_offset = (grid_num - 1) * max_grid_size;
        int last_grid_size = m - last_grid_offset;
        tsgemm<<<grid_dim, block_dim>>>(last_grid_size, n, &A[last_grid_offset], lda,
                                        &d_work1[(grid_num - 1) * n], ldwork1, &A[last_grid_offset], lda);

    } else {
        assert((m % block_size) % n == 0);
        assert(block_size % n == 0);

        int reduction_time =
            ceil((log(m) - log(n)) / (log(block_size) - log(n)));
        // printf("size %d, reduction_time: %d\n", m, reduction_time);
        int share_memory_size = reduction_time * block_size * n * sizeof(T);

        CUDA_CHECK(cudaFuncSetAttribute(
            householder_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            share_memory_size));

        dim3 block_dim(32,
                       32);  // if change block_dim, also change acc_per_thread
                             // and q_per_thread mannually in kernelQR.h
        int block_num = (m + block_size - 1) / block_size;
        householder_kernel<T><<<block_num, block_dim, share_memory_size>>>(
            block_size, m, n, A, lda, R, ldr, d_work2, ldwork2);
    }
}

template void tsqr<double>(cublasHandle_t cublas_handle,
                                     int block_size, int m, int n, double *A,
                                     int lda, double *R, int ldr,
                                     double *d_work1, int lwork1,
                                     double *d_work2, int lwork2);
