#pragma once

#include "TallShinnyQR.h"

#define N_B 32

template <typename T>
__global__ void gemm_blockqr(int m, int n, int k, T alpha, int trans_A, T *A,
                             const int lda, T *B, const int ldb, T beta, T *C,
                             const int ldc) {
    const int grid_dim_x = gridDim.x, grid_dim_y = gridDim.y;
    const int block_dim_x = blockDim.x, block_dim_y = blockDim.y;
    const int block_idx_x = blockIdx.x, block_idx_y = blockIdx.y;
    const int thread_idx_x = threadIdx.x, thread_idx_y = threadIdx.y;

    const int num_col =
        (n + grid_dim_y * block_dim_y - 1) / grid_dim_y * block_dim_y;
    const int num_row =
        (m + grid_dim_x * block_dim_x - 1) / grid_dim_x * block_dim_x;

    if (trans_A == 0) {
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
                C[row_idx + col_idx * ldc] =
                    alpha * sum + beta * C[row_idx + col_idx * ldc];
            }
        }
    } else {  // trans_A == 1
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
                    sum += A[row_idx * lda + i] * B[i + col_idx * ldb];
                }
                C[row_idx + col_idx * ldc] =
                    alpha * sum + beta * C[row_idx + col_idx * ldc];
            }
        }
    }
}
template __global__ void gemm_blockqr<double>(int m, int n, int k, double alpha,
                                              int trans_A, double *A,
                                              const int lda, double *B,
                                              const int ldb, double beta,
                                              double *C, const int ldc);
template __global__ void gemm_blockqr<float>(int m, int n, int k, float alpha,
                                             int trans_A, float *A,
                                             const int lda, float *B,
                                             const int ldb, float beta,
                                             float *C, const int ldc);

template <typename T>
void qr(int m, int n, T *A, int lda, T *R, int ldr, T *d_work, int ldwork) {
    dim3 block_dim_gemm{32, 32};
    if (n <= N_B) {
        tsqr<T>(m, n, A, lda, R, ldr, d_work, ldwork);
    } else {
        assert(n % 2 == 0);
        assert((n / 2) % N_B == 0);

        int n1 = n / 2;
        T *A1 = A, *A2 = A + n1 * lda;
        T *R11 = R, *R12 = R + n1 * ldr, *R22 = R + n1 * ldr + n1;
        dim3 gridDim1{(n1 + block_dim_gemm.x - 1) / block_dim_gemm.x,
                      (n1 + block_dim_gemm.y - 1) / block_dim_gemm.y};
        dim3 gridDim2 = {NUM_SM, 1};

        qr<T>(m, n1, A1, lda, R11, ldr, d_work, ldwork);
        gemm_blockqr<T><<<gridDim1, block_dim_gemm>>>(n1, n1, m, 1, 1, A1, lda,
                                                      A2, lda, 0.0, R12, ldr);
        gemm_blockqr<T><<<gridDim2, block_dim_gemm>>>(
            m, n1, n1, -1.0, 0, A1, lda, R12, ldr, 1.0, A2, lda);
        qr<T>(m, n1, A2, lda, R22, ldr, d_work, ldwork);
    }
}
template void qr<double>(int m, int n, double *A, int lda, double *R, int ldr,
                         double *d_work, int ldwork);
template void qr<float>(int m, int n, float *A, int lda, float *R, int ldr,
                        float *d_work, int ldwork);