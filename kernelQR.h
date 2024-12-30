#include <cuda_fp16.h>

#define BLOCK_SIZE 256
#define BLOCK_DIM_Y 16
#define BLOCK_DIM_X 32
#define NUM_Q_ROW 8
#define NUM_Q_COL 2  // (n + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
#define MAX_N 32

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
static __inline__ __device__ T warpAllReduceSum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void tsqr_kernel(int m, int n, T *A, int lda, T *R, int ldr) {
    // 创建shared memory，让整个block的线程能够进行数据共享
    shared_memory<T> shared;
    T *shared_A = shared.get_pointer();

    int ldsa = BLOCK_SIZE;

    const int thread_idx_x = threadIdx.x;
    const int thread_idx_y = threadIdx.y;
    const int block_idx_x = blockIdx.x;
    const int block_dim_x = BLOCK_DIM_X;
    const int block_dim_y = BLOCK_DIM_Y;

    A = A + block_idx_x * BLOCK_SIZE;
    R = R + block_idx_x * n;

    // 每个线程处理的数据个数
    int num_data_col = (n + block_dim_y - 1) / block_dim_y;

    T acc[NUM_Q_ROW];

    // 假定n=N=32，每一个线程拷贝2列
#pragma unroll
    for (int k = 0; k < NUM_Q_ROW; ++k) {
        int row_idx = thread_idx_x + k * block_dim_x;
        for (int h = 0; h < num_data_col; ++h) {
            int col_idx = thread_idx_y + h * block_dim_y;
            if (col_idx < n) {
                shared_A[row_idx + col_idx * ldsa] = A[row_idx + col_idx * lda];
            }
        }
    }

    // 需要进行整个block的同步，应该只需要1个lane进行同步就行---需要思考一下
    // __syncwarp();

    T q[NUM_Q_ROW];
    // 进行HouseHolder分解，先计算HouseHolder向量
    // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1));
    // 3、u=u/sqrt(abs(u(1)))
    for (int cols = 0; cols < n; cols++) {
        // 先计算HouseHolder向量
        // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1));
        // 3、u=u/sqrt(abs(u(1)))
        T nu = 0.0;
        if (thread_idx_y == cols % block_dim_y) {
            // 0.求normx
            // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
            for (int k = 0; k < NUM_Q_ROW; k++) {
                acc[k] = 0.0;
                int row_idx = thread_idx_x + k * block_dim_x;
                // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                if (row_idx >= cols) {
                    q[k] = shared_A[row_idx + cols * ldsa];
                    acc[k] = q[k] * q[k];
                }
                nu += acc[k];
            }

            // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
            T norm_x_squre = warpAllReduceSum(nu);
            T norm_x = sqrt(norm_x_squre);

            // 1、求u=x/norm(x);
            T scale = 1.0 / norm_x;
#pragma unroll
            for (int k = 0; k < NUM_Q_ROW; k++) {
                int row_idx = thread_idx_x + k * block_dim_x;
                if (row_idx >= cols) {
                    q[k] *= scale;
                }
            }

            int thread_idx = cols % block_dim_x;
            int thread_off = cols / block_dim_x;
            T u1 = 0;
            if (thread_idx_x == thread_idx) {
                q[thread_off] += (q[thread_off] >= 0) ? 1 : -1;
                u1 = q[thread_off];
                R[cols + cols * ldr] = (u1 >= 0) ? -norm_x : norm_x;
            }
            u1 = __shfl_sync(0xFFFFFFFF, u1, thread_idx);

            // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
            scale = 1 / (sqrt(abs(u1)));
#pragma unroll
            for (int k = 0; k < NUM_Q_ROW; k++) {
                int row_idx = thread_idx_x + k * block_dim_x;
                if (row_idx >= cols) {
                    shared_A[row_idx + cols * ldsa] = q[k] * scale;
                }
            }
        }

        __syncthreads();

        // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
        // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
        // 每个线程按列需要处理多个列
        for (int h = 0; h < num_data_col; h++) {
            int opCols = thread_idx_y + h * block_dim_y;

            // 只更新当前列后面的列
            if (cols < opCols && opCols < n) {
                nu = 0.0;
                // 先计算u'x
#pragma unroll
                for (int k = 0; k < NUM_Q_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = thread_idx_x + k * block_dim_x;
                    // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                    if (row_idx >= cols) {
                        q[k] = shared_A[row_idx + cols * ldsa];
                        acc[k] = q[k] * shared_A[row_idx + opCols * ldsa];
                    }
                    nu += acc[k];
                }
                T utx = warpAllReduceSum(nu);

                // 计算x-uu'x
#pragma unroll
                for (int k = 0; k < NUM_Q_ROW; k++) {
                    int row_idx = thread_idx_x + k * block_dim_x;
                    // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                    if (row_idx >= cols) {
                        shared_A[row_idx + opCols * ldsa] -= utx * q[k];
                    }
                }
            }
        }
    }

    __syncthreads();
    // 此时已经完成HouseHolder更新，在AA中存放着HouseHolder向量和R矩阵的上三角部分,RR中存放在对角线元素

    // 获得R矩阵，将AA的上三角部分拷贝到R中
    // 以R矩阵来进行循环
    int rRowDataNum = (n + (block_dim_x - 1)) / block_dim_x;
    for (int h = 0; h < num_data_col; h++) {
        int opCols = thread_idx_y + h * block_dim_y;

        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < rRowDataNum; k++) {
            int row_idx = thread_idx_x + k * block_dim_x;
            if (row_idx < opCols) {
                R[row_idx + opCols * ldr] = shared_A[row_idx + opCols * ldsa];
                shared_A[row_idx + opCols * ldsa] = 0.0;
            } else if (row_idx > opCols) {
                R[row_idx + opCols * ldr] = 0.0;
            }
        }
    }

    // 来求Q，使用的方法是Q=(I-uu')Q, 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q
    // q表示是Q矩阵的1列
    for (int h = 0; h < num_data_col; h++) {
        // 1、构造出每个线程需要处理的Q矩阵的一列q的一部分
        int opCols = thread_idx_y + h * block_dim_y;

        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < NUM_Q_ROW; k++) {
            if (thread_idx_x + k * block_dim_x == opCols) {
                q[k] = 1.0;
            } else {
                q[k] = 0.0;
            }
        }

        __syncwarp();

        for (int cols = n - 1; cols >= 0; cols--) {
            // 这个判断没有问题，很经典，实际上不带这个判断也是正确的。这个判断是利用矩阵特点对矩阵乘法的一种优化
            // 因为Q_k-1=(I-u_k-1*u_k-1')*Q_k-2也是一个左上角是单位矩阵，右下角是一个k-1xk-1的矩阵，其他部分都是0；
            // 而I-uk*uk'也是一个左上角是单位矩阵，右下角是一个kxk的矩阵，其他部分为0；所以两者相乘只影响后面大于等于k的列
            if (opCols >= cols) {
                // 2、计算u'q
                T nu = 0.0;
#pragma unroll
                for (int k = 0; k < NUM_Q_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = thread_idx_x + k * block_dim_x;
                    acc[k] = shared_A[row_idx + cols * ldsa] * q[k];
                    nu += acc[k];
                }

                T utq = warpAllReduceSum(nu);

                // 3.计算q-uu'q
#pragma unroll
                for (int k = 0; k < NUM_Q_ROW; k++) {
                    int row_idx = thread_idx_x + k * block_dim_x;
                    q[k] -= utq * shared_A[row_idx + cols * ldsa];
                }

                __syncwarp();
            }
        }

        // 4.把计算出来的q拷贝到A中
#pragma unroll
        for (int k = 0; k < NUM_Q_ROW; k++) {
            int row_idx = thread_idx_x + k * block_dim_x;
            A[row_idx + opCols * lda] = q[k];
        }
    }
}

template __global__ void tsqr_kernel<float>(int m, int n, float *A, int lda,
                                            float *R, int ldr);
template __global__ void tsqr_kernel<double>(int m, int n, double *A, int lda,
                                             double *R, int ldr);