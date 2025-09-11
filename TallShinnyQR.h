#pragma once
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>

#include <cassert>

#define TSQR_BLOCK_SIZE 256
#define TSQR_BLOCK_DIM_Y 4
#define TSQR_BLOCK_DIM_X 64
#define TSQR_NUM_DATA_ROW 4

template <typename T>
struct shared_memory;
template <>
struct shared_memory<float> {
    __device__ static float *get_pointer() {
        extern __shared__ float shared_mem_float[];
        return shared_mem_float;
    }
};
template <>
struct shared_memory<double> {
    __device__ static double *get_pointer() {
        extern __shared__ double shared_mem_double[];
        return shared_mem_double;
    }
};

#pragma once
template <typename T>
static __inline__ __device__ T warp_all_reduce_sum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor(val, mask);
    }
    return val;
}

template <typename T>
__global__ void tsqr_kernel(int m, int n, T* A, int lda, T* R, int ldr) {
    shared_memory<T> shared;
    T* shared_A = shared.get_pointer();
    int ldsa = TSQR_BLOCK_SIZE;

    const int thread_idx_x = threadIdx.x;
    const int thread_idx_y = threadIdx.y;
    const int block_idx_x = blockIdx.x;

    int block_size = min(TSQR_BLOCK_SIZE, m - block_idx_x * TSQR_BLOCK_SIZE);

    A = A + block_idx_x * TSQR_BLOCK_SIZE;
    R = R + block_idx_x * n;

    int num_data_col = (n + TSQR_BLOCK_DIM_Y - 1) / TSQR_BLOCK_DIM_Y;

    T acc[TSQR_NUM_DATA_ROW];

#pragma unroll
    for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
        int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
        if (row_idx < block_size) {
            for (int h = 0; h < num_data_col; ++h) {
                int col_idx = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
                if (col_idx < n) {
                    shared_A[row_idx + col_idx * ldsa] =
                        A[row_idx + col_idx * lda];
                }
            }
        }
    }

    __syncthreads();

    T q[TSQR_NUM_DATA_ROW];

    for (int cols = 0; cols < n; cols++) {
        T nu = 0.0;
        if (thread_idx_y == cols % TSQR_BLOCK_DIM_Y) {
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                acc[k] = 0.0;
                int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    q[k] = shared_A[row_idx + cols * ldsa];
                    acc[k] = q[k] * q[k];
                }
                nu += acc[k];
            }

            T norm_x_square = warp_all_reduce_sum(nu);
            T norm_x = sqrt(norm_x_square);

            constexpr T epsilon = std::is_same<T, double>::value ? 1e-12 : 1e-7;

            if (norm_x > epsilon) {
                T scale = 1.0 / norm_x;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        q[k] *= scale;
                    }
                }

                int thread_idx = cols % TSQR_BLOCK_DIM_X;
                int thread_off = cols / TSQR_BLOCK_DIM_X;
                T u1 = 0;
                if (thread_idx_x == thread_idx) {
                    q[thread_off] += (q[thread_off] >= 0) ? 1.0 : -1.0;
                    u1 = q[thread_off];
                    R[cols + cols * ldr] = (u1 >= 0) ? -norm_x : norm_x;
                }
                u1 = __shfl_sync(u1, thread_idx);

                scale = 1.0 / (sqrt(abs(u1)));
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + cols * ldsa] = q[k] * scale;
                    }
                }
            } else {
                int thread_idx = cols % TSQR_BLOCK_DIM_X;
                if (thread_idx_x == thread_idx) {
                    R[cols + cols * ldr] = 0.0;
                }
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + cols * ldsa] = 0.0;
                    }
                }
            }
        }

        __syncthreads();

        for (int h = 0; h < num_data_col; h++) {
            int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
            if (cols < opCols && opCols < n) {
                nu = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        q[k] = shared_A[row_idx + cols * ldsa];
                        acc[k] = q[k] * shared_A[row_idx + opCols * ldsa];
                    }
                    nu += acc[k];
                }
                T utx = warp_all_reduce_sum(nu);

#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + opCols * ldsa] -= utx * q[k];
                    }
                }
            }
        }
    }

    __syncthreads();

    int rRowDataNum = (n + (TSQR_BLOCK_DIM_X - 1)) / TSQR_BLOCK_DIM_X;
    for (int h = 0; h < num_data_col; h++) {
        int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < rRowDataNum; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            if (row_idx < opCols && row_idx < n) {
                R[row_idx + opCols * ldr] = shared_A[row_idx + opCols * ldsa];
                shared_A[row_idx + opCols * ldsa] = 0.0;
            }
            if (row_idx > opCols && row_idx < n) {
                R[row_idx + opCols * ldr] = 0.0;
            }
        }
    }

    for (int h = 0; h < num_data_col; h++) {
        int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            q[k] = (row_idx == opCols) ? 1.0 : 0.0;
        }
        __syncwarp();

        for (int cols = n - 1; cols >= 0; cols--) {
            if (opCols >= cols) {
                T nu = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        acc[k] = shared_A[row_idx + cols * ldsa] * q[k];
                        nu += acc[k];
                    }
                }
                T utq = warp_all_reduce_sum(nu);

#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        q[k] -= utq * shared_A[row_idx + cols * ldsa];
                    }
                }
                __syncwarp();
            }
        }

#pragma unroll
        for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            if (row_idx < block_size) {
                A[row_idx + opCols * lda] = q[k];
            }
        }
    }
}
template __global__ void tsqr_kernel<float>(int m, int n, float *A, int lda,
                                            float *R, int ldr);
template __global__ void tsqr_kernel<double>(int m, int n, double *A, int lda,
                                             double *R, int ldr);

template <typename T>
void tsqr_func(cublasHandle_t cublas_handle, int share_memory_size, int m,
               int n, T *A, int lda, T *R, int ldr, T *work, int ldwork);
template <>
void tsqr_func(cublasHandle_t cublas_handle, int share_memory_size, int m,
               int n, float *A, int lda, float *R, int ldr, float *work,
               int ldwork) {
    // 一个block最大为32x32，一个block中的thread可以使用共享内存进行通信，
    //  所以使用一个block处理一个最大为<TSQR_BLOCK_SIZE,N>的矩阵块，并对它进行QR分解
    dim3 blockDim(TSQR_BLOCK_DIM_X, TSQR_BLOCK_DIM_Y);

    // 1.如果m<=TSQR_BLOCK_SIZE,就直接调用核函数进行QR分解
    if (m <= TSQR_BLOCK_SIZE) {
        // 调用核函数进行QR分解
        // 分解后A矩阵中存放的是Q矩阵，R矩阵中存放的是R矩阵
        tsqr_kernel<<<1, blockDim, share_memory_size>>>(m, n, A, lda, R, ldr);
        cudaDeviceSynchronize();
        return;
    }

    // 2.使用按列进行分段的方式进行QR分解
    // 2.1 把瘦高矩阵进行按列分段
    int blockNum = (m + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;

    // 2.2直接创建这么多个核函数进行QR分解,A中存放Q, work中存放R
    tsqr_kernel<<<blockNum, blockDim, share_memory_size>>>(m, n, A, lda, work,
                                                           ldwork);

    // 2.3再对R进行QR分解,也就是对work进行递归调用此函数
    tsqr_func(cublas_handle, share_memory_size, blockNum * n, n, work, ldwork,
              R, ldr, work + n * ldwork, ldwork);

    // 3.求出最终的Q，存放到A中
    // 注意这里使用了一个batch乘积的方法，是一个非常有趣的思想,需要结合瘦高矩阵的分块矩阵理解，非常有意思
    float tone = 1.0, tzero = 0.0;
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              TSQR_BLOCK_SIZE, n, n, &tone, A, lda,
                              TSQR_BLOCK_SIZE, work, ldwork, n, &tzero, A, lda,
                              TSQR_BLOCK_SIZE, m / TSQR_BLOCK_SIZE);

    // 3.2如果m/M还有剩余的话，还需要计算最后一个块的Q进行乘法计算，才能得到最终的Q
    int mm = m % TSQR_BLOCK_SIZE;
    if (0 < mm) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, n, &tone,
                    A + (m - mm), lda, work + (m / TSQR_BLOCK_SIZE * n), ldwork,
                    &tzero, A + (m - mm), lda);
    }
}
template <>
void tsqr_func(cublasHandle_t cublas_handle, int share_memory_size, int m,
               int n, double *A, int lda, double *R, int ldr, double *work,
               int ldwork) {
    // 一个block最大为32x32，一个block中的thread可以使用共享内存进行通信，
    //  所以使用一个block处理一个最大为<TSQR_BLOCK_SIZE,N>的矩阵块，并对它进行QR分解
    dim3 blockDim(TSQR_BLOCK_DIM_X, TSQR_BLOCK_DIM_Y);

    // 1.如果m<=TSQR_BLOCK_SIZE,就直接调用核函数进行QR分解
    if (m <= TSQR_BLOCK_SIZE) {
        // 调用核函数进行QR分解
        // 分解后A矩阵中存放的是Q矩阵，R矩阵中存放的是R矩阵
        tsqr_kernel<<<1, blockDim, share_memory_size>>>(m, n, A, lda, R, ldr);
        cudaDeviceSynchronize();
        return;
    }

    // 2.使用按列进行分段的方式进行QR分解
    // 2.1 把瘦高矩阵进行按列分段
    int blockNum = (m + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;

    // 2.2直接创建这么多个核函数进行QR分解,A中存放Q, work中存放R
    tsqr_kernel<<<blockNum, blockDim, share_memory_size>>>(m, n, A, lda, work,
                                                           ldwork);

    // 2.3再对R进行QR分解,也就是对work进行递归调用此函数
    tsqr_func(cublas_handle, share_memory_size, blockNum * n, n, work, ldwork,
              R, ldr, work + n * ldwork, ldwork);

    // 3.求出最终的Q，存放到A中
    // 注意这里使用了一个batch乘积的方法，是一个非常有趣的思想,需要结合瘦高矩阵的分块矩阵理解，非常有意思
    double tone = 1.0, tzero = 0.0;
    cublasDgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              TSQR_BLOCK_SIZE, n, n, &tone, A, lda,
                              TSQR_BLOCK_SIZE, work, ldwork, n, &tzero, A, lda,
                              TSQR_BLOCK_SIZE, m / TSQR_BLOCK_SIZE);

    // 3.2如果m/M还有剩余的话，还需要计算最后一个块的Q进行乘法计算，才能得到最终的Q
    int mm = m % TSQR_BLOCK_SIZE;
    if (0 < mm) {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, n, &tone,
                    A + (m - mm), lda, work + (m / TSQR_BLOCK_SIZE * n), ldwork,
                    &tzero, A + (m - mm), lda);
    }
}

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T>
void tsqr(cublasHandle_t cublas_handle, int m, int n, T *A, int lda, T *R,
          int ldr, T *work, int ldwork) {
    assert(m >= n);
    assert(m % n == 0);
    // assert(m % TSQR_BLOCK_SIZE == 0);
    // assert(n % TSQR_BLOCK_DIM_Y == 0);
    assert(((m % TSQR_BLOCK_SIZE) % n) == 0);
    assert(TSQR_BLOCK_SIZE % TSQR_BLOCK_DIM_X == 0);
    assert(TSQR_BLOCK_DIM_X * TSQR_NUM_DATA_ROW == TSQR_BLOCK_SIZE);

    int share_memory_size = TSQR_BLOCK_SIZE * n * sizeof(T);
    cudaFuncSetAttribute(tsqr_kernel<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         share_memory_size);

    tsqr_func(cublas_handle, share_memory_size, m, n, A, lda, R, ldr, work,
              ldwork);
}

template void tsqr<float>(cublasHandle_t cublas_handle, int m, int n, float *A,
                          int lda, float *R, int ldr, float *work, int ldwork);
template void tsqr<double>(cublasHandle_t cublas_handle, int m, int n,
                           double *A, int lda, double *R, int ldr, double *work,
                           int ldwork);