/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cuComplex.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <library_types.h>

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// CUDA API error checking
#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t err_ = (err);                                            \
        if (err_ != cudaSuccess) {                                           \
            printf("CUDA error %d at %s:%d\n%s\n", err_, __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
            throw std::runtime_error("CUDA error");                          \
        }                                                                    \
    } while (0)

// CUDA API error checking
#define CUDA_CHECK_LAST_ERROR()                                              \
    do {                                                                     \
        cudaError_t err_ = (cudaGetLastError());                             \
        if (err_ != cudaSuccess) {                                           \
            printf("CUDA error %d at %s:%d\n%s\n", err_, __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
            throw std::runtime_error("CUDA error");                          \
        }                                                                    \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                   \
    do {                                                                      \
        cusolverStatus_t err_ = (err);                                        \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusolver error");                       \
        }                                                                     \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                   \
    do {                                                                    \
        cublasStatus_t err_ = (err);                                        \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                       \
        }                                                                   \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                   \
    do {                                                                      \
        cusparseStatus_t err_ = (err);                                        \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusparse error");                       \
        }                                                                     \
    } while (0)

// memory alignment
#define ALIGN_TO(A, B) (((A + B - 1) / B) * B)

// device memory pitch alignment
static const size_t device_alignment = 32;

void print_device_info() {
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << "reservedSharedMemPerBlock: "
              << (device_prop.reservedSharedMemPerBlock >> 10) << " KB "
              << std::endl;
    std::cout << "sharedMemPerBlock: " << (device_prop.sharedMemPerBlock >> 10)
              << " KB " << std::endl;
    std::cout << "sharedMemPerBlockOptin: "
              << (device_prop.sharedMemPerBlockOptin >> 10) << " KB "
              << std::endl;
    std::cout << "sharedMemPerMultiprocessor: "
              << (device_prop.sharedMemPerMultiprocessor >> 10) << " KB "
              << std::endl;
    std::cout << std::endl;
}

template <typename T>
void print_device_matrix(T *dA, long ldA, long rows, long cols) {
    T matrix;

    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < cols; j++) {
            cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T),
                       cudaMemcpyDeviceToHost);
            printf("%9.6f ", matrix);
        }
        printf("\n");
    }
}
template void print_device_matrix(double *dA, long ldA, long rows, long cols);
template void print_device_matrix(float *dA, long ldA, long rows, long cols);

template <typename T>
bool all_close(T const *A, T const *A_ref, size_t m, size_t n, size_t lda,
               T abs_tol, T rel_tol) {
    bool status{true};
    for (size_t j{0U}; j < n; ++j) {
        for (size_t i{0U}; i < m; ++i) {
            T const A_val{static_cast<T>(A[i + j * lda])};
            T const A_ref_val{static_cast<T>(A_ref[i + j * lda])};
            T const diff{A_val - A_ref_val};
            T const diff_val{std::abs(diff)};
            if (std::isnan(diff_val) ||
                diff_val >
                    std::max(static_cast<T>(abs_tol),
                             static_cast<T>(std::abs(A_ref_val)) * rel_tol)) {
                std::cout << "A[" << i << ", " << j << "] = " << A_val
                          << " A_ref[" << i << ", " << j << "] = " << A_ref_val
                          << " Abs Diff: " << diff_val
                          << " Abs Diff Threshold: " << static_cast<T>(abs_tol)
                          << " Rel->Abs Diff Threshold: "
                          << static_cast<T>(
                                 static_cast<T>(std::abs(A_ref_val)) * rel_tol)
                          << std::endl;
                status = false;
                return status;
            }
        }
    }
    return status;
}
template bool all_close(double const *A, double const *A_ref, size_t m,
                        size_t n, size_t lda, double abs_tol, double rel_tol);
template bool all_close(float const *A, float const *A_ref, size_t m, size_t n,
                        size_t lda, float abs_tol, float rel_tol);

template <typename T>
__global__ void init_identity_matrix(T *matrix, int ldm, int m, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j)
                    matrix[i + j * ldm] = 1;
                else
                    matrix[i + j * ldm] = 0;
            }
        }
    }
}
template __global__ void init_identity_matrix<double>(double *matrix, int ldm,
                                                      int m, int n);
template __global__ void init_identity_matrix<float>(float *matrix, int ldm,
                                                     int m, int n);

template <typename T>
__global__ void copy_matrix(int m, int n, T *dst, int ldst, T *src, int ldsrc) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            dst[i + j * ldst] = src[i + j * ldsrc];
        }
    }
}
template __global__ void copy_matrix<double>(int m, int n, double *dst,
                                             int ldst, double *src, int ldsrc);
template __global__ void copy_matrix<float>(int m, int n, float *dst, int ldst,
                                            float *src, int ldsrc);

template <typename T>
T get_matrix_2_norm(cusolverDnHandle_t cusolverH, int m, int n, T *A, int lda);
template <>
double get_matrix_2_norm(cusolverDnHandle_t cusolverH, int m, int n, double *A,
                         int lda) {
    double s = 0;
    const int ldu = m;   // ldu >= m
    const int ldvt = n;  // ldvt >= n if jobu = 'A'
    int ldwork = 0;
    int info_gpu = 0;

    int *devInfo = nullptr;
    double *d_work = nullptr;
    double *d_S = nullptr;
    double *d_U = nullptr;  /* left singular vectors */
    double *d_VT = nullptr; /* right singular vectors */

    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &ldwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work),
                          ldwork * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), n * sizeof(double)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_U), ldu * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT),
                          ldvt * n * sizeof(double)));

    CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, 'N', 'N', m, n, A, lda, d_S, d_U,
                                    ldu, d_VT, ldvt, d_work, ldwork, nullptr,
                                    devInfo));
    CUDA_CHECK(
        cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_gpu < 0) {
        std::printf("%d-th parameter is wrong \n", -info_gpu);
        exit(1);
    } else if (info_gpu > 0) {
        std::printf("WARNING: info = %d : gesvd does not converge \n",
                    info_gpu);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(&s, d_S, sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_VT));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));

    return s;
}
template <>
float get_matrix_2_norm(cusolverDnHandle_t cusolverH, int m, int n, float *A,
                        int lda) {
    float s = 0;
    const int ldu = m;   // ldu >= m
    const int ldvt = n;  // ldvt >= n if jobu = 'A'
    int ldwork = 0;
    int info_gpu = 0;

    int *devInfo = nullptr;
    float *d_work = nullptr;
    float *d_S = nullptr;
    float *d_U = nullptr;  /* left singular vectors */
    float *d_VT = nullptr; /* right singular vectors */

    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &ldwork));

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), ldwork * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), n * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_U), ldu * m * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_VT), ldvt * n * sizeof(float)));

    CUSOLVER_CHECK(cusolverDnSgesvd(cusolverH, 'N', 'N', m, n, A, lda, d_S, d_U,
                                    ldu, d_VT, ldvt, d_work, ldwork, nullptr,
                                    devInfo));
    CUDA_CHECK(
        cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_gpu < 0) {
        std::printf("%d-th parameter is wrong \n", -info_gpu);
        exit(1);
    } else if (info_gpu > 0) {
        std::printf("WARNING: info = %d : gesvd does not converge \n",
                    info_gpu);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(&s, d_S, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_VT));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));

    return s;
}

template <typename T>
void check_QR_accuracy(int m, int n, T *d_A, int ldq, T *R, int ldr,
                       std::vector<T> &A);
template <>
void check_QR_accuracy<double>(int m, int n, double *d_A, int lda, double *d_R,
                               int ldr, std::vector<double> &A) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    double one = 1, zero = 0, minus_one = -1;
    const int ldqtq = n;
    const int ldqr = m;
    double *d_QTQ = nullptr;
    double *d_QR = nullptr;

    std::vector<double> A_from_gpu(m * n, 0);

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_QTQ), sizeof(double) * n * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_QR), sizeof(double) * m * n));

    // QR = Q * R
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one,
                             d_A, lda, d_R, ldr, &zero, d_QR, ldqr));

    // move QR to host memory
    CUDA_CHECK(cudaMemcpy(A_from_gpu.data(), d_QR,
                          sizeof(double) * A_from_gpu.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    double abs_tol = 1.0e-4, rel_tol = 1.0e-5;
    // compare QR with original A
    if (!all_close(A_from_gpu.data(), A.data(), m, n, lda, abs_tol, rel_tol)) {
        std::cout << "Error: tsqr" << std::endl;
        exit(-1);
    }

    // d_QTQ = I - Q^T * Q
    init_identity_matrix<<<1, 1>>>(d_QTQ, ldqtq, n, n);
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &minus_one, d_A, lda, d_A, lda, &one, d_QTQ,
                             ldqtq));
    double QTQ_2_norm = get_matrix_2_norm(cusolverH, n, n, d_QTQ, ldqtq);

    // d_QR = A
    CUDA_CHECK(cudaMemcpy(d_QR, A.data(), sizeof(double) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double A_2_norm = get_matrix_2_norm(cusolverH, m, n, d_QR, ldqr);

    // d_QR = A - QR
    CUDA_CHECK(cudaMemcpy(d_QR, A.data(), sizeof(double) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &minus_one, d_A, lda, d_R, ldr, &one, d_QR, ldqr));
    double QR_2_norm = get_matrix_2_norm(cusolverH, m, n, d_QR, ldqr);

    printf("|A-QR|/|A| = %.17f, |I-Q^TQ| / n = %.17f\n", QR_2_norm / A_2_norm,
           QTQ_2_norm/ n);

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaFree(d_QTQ));
    CUDA_CHECK(cudaFree(d_QR));
}
template <>
void check_QR_accuracy<float>(int m, int n, float *d_A, int lda, float *d_R,
                              int ldr, std::vector<float> &A) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    float one = 1, zero = 0, minus_one = -1;
    const int ldqtq = n;
    const int ldqr = m;
    float *d_QTQ = nullptr;
    float *d_QR = nullptr;

    std::vector<float> A_from_gpu(m * n, 0);

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_QTQ), sizeof(float) * n * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_QR), sizeof(float) * m * n));

    // QR = Q * R
    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one,
                             d_A, lda, d_R, ldr, &zero, d_QR, ldqr));

    // move QR to host memory
    CUDA_CHECK(cudaMemcpy(A_from_gpu.data(), d_QR,
                          sizeof(float) * A_from_gpu.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    float abs_tol = 1.0e-4, rel_tol = 1.0e-5;
    // compare QR with original A
    if (!all_close(A_from_gpu.data(), A.data(), m, n, lda, abs_tol, rel_tol)) {
        std::cout << "Error: tsqr" << std::endl;
        exit(-1);
    }

    // d_QTQ = I - Q^T * Q
    init_identity_matrix<<<1, 1>>>(d_QTQ, ldqtq, n, n);
    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &minus_one, d_A, lda, d_A, lda, &one, d_QTQ,
                             ldqtq));
    float QTQ_2_norm = get_matrix_2_norm(cusolverH, n, n, d_QTQ, ldqtq);

    // d_QR = A
    CUDA_CHECK(cudaMemcpy(d_QR, A.data(), sizeof(float) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    float A_2_norm = get_matrix_2_norm(cusolverH, m, n, d_QR, ldqr);

    // d_QR = A - QR
    CUDA_CHECK(cudaMemcpy(d_QR, A.data(), sizeof(float) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &minus_one, d_A, lda, d_R, ldr, &one, d_QR, ldqr));
    float QR_2_norm = get_matrix_2_norm(cusolverH, m, n, d_QR, ldqr);

    printf("|A-QR|/|A| = %.17f, |I-Q^TQ| / n = %.17f\n", QR_2_norm / A_2_norm,
           QTQ_2_norm / n);

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaFree(d_QTQ));
    CUDA_CHECK(cudaFree(d_QR));
}
