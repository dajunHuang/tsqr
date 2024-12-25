#pragma once

#include <cusolverDn.h>

#include <iostream>

#include "kernelQR.h"
#include "utils.h"

template <typename T>
void tsqr_func(cublasHandle_t cublas_handle, cudaDataType_t cuda_data_type,
               cublasComputeType_t cublas_compute_type, int share_memory_size,
               int m, int n, T *A, int lda, T *R, int ldr, T *work,
               int ldwork) {
    // 一个block最大为32x32，一个block中的thread可以使用共享内存进行通信，
    //  所以使用一个block处理一个最大为<BLOCK_SIZE,N>的矩阵块，并对它进行QR分解
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);

    // 1.如果m<=BLOCK_SIZE,就直接调用核函数进行QR分解
    if (m <= BLOCK_SIZE) {
        // 调用核函数进行QR分解
        // 分解后A矩阵中存放的是Q矩阵，R矩阵中存放的是R矩阵
        tsqr_kernel<T>
            <<<1, blockDim, share_memory_size>>>(m, n, A, lda, R, ldr);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    // 2.使用按列进行分段的方式进行QR分解
    // 2.1 把瘦高矩阵进行按列分段
    int blockNum = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 2.2直接创建这么多个核函数进行QR分解,A中存放Q, work中存放R
    tsqr_kernel<T>
        <<<blockNum, blockDim, share_memory_size>>>(m, n, A, lda, work, ldwork);

    // 2.3再对R进行QR分解,也就是对work进行递归调用此函数
    tsqr_func<T>(cublas_handle, cuda_data_type, cublas_compute_type,
                 share_memory_size, blockNum * n, n, work, ldwork, R, ldr,
                 work + n * ldwork, ldwork);

    // 3.求出最终的Q，存放到A中
    // 注意这里使用了一个batch乘积的方法，是一个非常有趣的思想,需要结合瘦高矩阵的分块矩阵理解，非常有意思
    T tone = 1.0, tzero = 0.0;
    cublasGemmStridedBatchedEx(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK_SIZE, n, n, &tone, A,
        cuda_data_type, lda, BLOCK_SIZE, work, cuda_data_type, ldwork, n,
        &tzero, A, cuda_data_type, lda, BLOCK_SIZE, m / BLOCK_SIZE,
        cublas_compute_type, CUBLAS_GEMM_DEFAULT);

    // 3.2如果m/M还有剩余的话，还需要计算最后一个块的Q进行乘法计算，才能得到最终的Q
    int mm = m % BLOCK_SIZE;
    if (0 < mm) {
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, n, &tone,
                     A + (m - mm), cuda_data_type, lda,
                     work + (m / BLOCK_SIZE * n), cuda_data_type, ldwork,
                     &tzero, A + (m - mm), cuda_data_type, lda,
                     cublas_compute_type, CUBLAS_GEMM_DEFAULT);
    }
}
template void tsqr_func<float>(cublasHandle_t cublas_handle,
                               cudaDataType_t cuda_data_type,
                               cublasComputeType_t cublas_compute_type,
                               int share_memory_size, int m, int n, float *A,
                               int lda, float *R, int ldr, float *work,
                               int ldwork);
template void tsqr_func<double>(cublasHandle_t cublas_handle,
                                cudaDataType_t cuda_data_type,
                                cublasComputeType_t cublas_compute_type,
                                int share_memory_size, int m, int n, double *A,
                                int lda, double *R, int ldr, double *work,
                                int ldwork);

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T>
void tsqr(cublasHandle_t cublas_handle, int m, int n, T *A, int lda, T *R,
          int ldr, T *work, int ldwork) {
    if (0 != (m % BLOCK_SIZE) % n) {
        std::cout << "hou_tsqr_panel QR the m%BLOCK_SIZE must be multis of n"
                  << std::endl;
        exit(1);
    }

    cudaDataType_t cuda_data_type;
    cublasComputeType_t cublas_compute_type;

    if (std::is_same<T, double>::value) {
        cuda_data_type = CUDA_R_64F;
        cublas_compute_type = CUBLAS_COMPUTE_64F;
    } else if (std::is_same<T, float>::value) {
        cuda_data_type = CUDA_R_32F;
        cublas_compute_type = CUBLAS_COMPUTE_32F;
    } else if (std::is_same<T, half>::value) {
        cuda_data_type = CUDA_R_16F;
        cublas_compute_type = CUBLAS_COMPUTE_16F;
    }

    int share_memory_size = BLOCK_SIZE * n * sizeof(T);
    CUDA_CHECK(cudaFuncSetAttribute(tsqr_kernel<T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    share_memory_size));

    tsqr_func(cublas_handle, cuda_data_type, cublas_compute_type,
              share_memory_size, m, n, A, lda, R, ldr, work, ldwork);
}

template void tsqr<float>(cublasHandle_t cublas_handle, int m, int n, float *A,
                          int lda, float *R, int ldr, float *work, int ldwork);
template void tsqr<double>(cublasHandle_t cublas_handle, int m, int n,
                           double *A, int lda, double *R, int ldr, double *work,
                           int ldwork);