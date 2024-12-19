#pragma once

#include "TallShinnyQR.h"

#define N_B 32

template <typename T>
void qr(int m, int n, T *A, int lda, T *R, int ldr, T *d_work, int ldwork);
template <>
void qr<double>(int m, int n, double *A, int lda, double *R, int ldr,
                double *d_work, int ldwork) {
    dim3 block_dim_gemm{32, 32};
    if (n <= N_B) {
        tsqr<double>(m, n, A, lda, R, ldr, d_work, ldwork);
    } else {
        assert(n % 2 == 0);
        assert((n / 2) % N_B == 0);

        int n1 = n / 2;
        double *A1 = A, *A2 = A + n1 * lda;
        double *R11 = R, *R12 = R + n1 * ldr, *R22 = R + n1 * ldr + n1;
        dim3 gridDim1{(n1 + block_dim_gemm.x - 1) / block_dim_gemm.x,
                      (n1 + block_dim_gemm.y - 1) / block_dim_gemm.y};
        dim3 gridDim2 = {(m + block_dim_gemm.x - 1) / block_dim_gemm.x,
                         (n1 + block_dim_gemm.y - 1) / block_dim_gemm.y};

        qr<double>(m, n1, A1, lda, R11, ldr, d_work, ldwork);
        tsgemm<double><<<gridDim1, block_dim_gemm>>>(n1, n1, m, 1, 1, A1, lda,
                                                     A2, lda, 0.0, R12, ldr);
        tsgemm<double><<<gridDim2, block_dim_gemm>>>(
            m, n1, n1, -1.0, 0, A1, lda, R12, ldr, 1.0, A2, lda);
        qr<double>(m, n1, A2, lda, R22, ldr, d_work, ldwork);
    }
}
template <>
void qr<float>(int m, int n, float *A, int lda, float *R, int ldr,
               float *d_work, int ldwork) {
    dim3 block_dim_gemm{32, 32};
    if (n <= N_B) {
        tsqr<float>(m, n, A, lda, R, ldr, d_work, ldwork);
    } else {
        assert(n % 2 == 0);
        assert((n / 2) % N_B == 0);

        int n1 = n / 2;
        float *A1 = A, *A2 = A + n1 * lda;
        float *R11 = R, *R12 = R + n1 * ldr, *R22 = R + n1 * ldr + n1;
        dim3 gridDim1{(n1 + block_dim_gemm.x - 1) / block_dim_gemm.x,
                      (n1 + block_dim_gemm.y - 1) / block_dim_gemm.y};
        dim3 gridDim2 = {(m + block_dim_gemm.x - 1) / block_dim_gemm.x,
                         (n1 + block_dim_gemm.y - 1) / block_dim_gemm.y};

        qr<float>(m, n1, A1, lda, R11, ldr, d_work, ldwork);
        tsgemm<float><<<gridDim1, block_dim_gemm>>>(n1, n1, m, 1, 1, A1, lda,
                                                    A2, lda, 0.0, R12, ldr);
        tsgemm<float><<<gridDim2, block_dim_gemm>>>(m, n1, n1, -1.0, 0, A1, lda,
                                                    R12, ldr, 1.0, A2, lda);
        qr<float>(m, n1, A2, lda, R22, ldr, d_work, ldwork);
    }
}