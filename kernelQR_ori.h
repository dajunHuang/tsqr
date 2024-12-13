#include <cuda_fp16.h>

#pragma once
template <typename T>
static __inline__ __device__ T warpAllReduceSum_ori(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T, long M, long N>
__global__ void my_hou_kernel_ori(long m, long n, T *A, long lda, T *R, long ldr) {
    // 1.求出本block处理的矩阵的尺寸
    long mm = min(m - blockIdx.x * M, M);

    // 理论上不会出现mm<=0的情况，这里只是错误处理
    if (0 >= mm) {
        return;
    }

    A = A + blockIdx.x * M;
    R = R + blockIdx.x * N;

    // printf("come 28\n");
    // // cudaDeviceSynchronize();
    // __syncthreads();

    // 目前的情况只会出现n<=N的情况，目前只处理n=N=32的情况
    long nn = min(N, n);

    // 2. 找到本线程的ID
    long i = threadIdx.x;
    long j = threadIdx.y;

    // 创建shared memory，让整个block的线程能够进行数据共享
    __shared__ T AA[M * N], RR[N];
    long ldaa = mm;

    // 每个线程处理的数据个数
    long rowDataNum = (mm + (blockDim.x - 1)) / blockDim.x;
    long colDataNum = (nn + (blockDim.y - 1)) / blockDim.y;

    // double acc[rowDataNum];
    T acc[8];

    // 假定n=N=32，每一个线程拷贝2列
    for (long k = 0; k < rowDataNum; k++) {
        if (i + k * blockDim.x < mm) {
            AA[i + k * blockDim.x + j * ldaa] = A[i + k * blockDim.x + j * lda];
            AA[i + k * blockDim.x + (j + 16) * ldaa] =
                A[i + k * blockDim.x + (j + 16) * lda];
        }
    }
    // if(blockIdx.x == 7 && i == 0 && j == 0) {
    //     printf("load to AA\n");
    //     for(int v = 0; v < mm; v++) {
    //         for(int l = 0; l < 32; l++) {
    //             printf("%9.6f ", AA[v + l * ldaa]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // 需要进行整个block的同步，应该只需要1个lane进行同步就行---需要思考一下
    __syncthreads();

    // 进行HouseHolder分解，先计算HouseHolder向量
    // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1));
    // 3、u=u/sqrt(abs(u(1)))
    for (long cols = 0; cols < nn; cols++) {
        // 先计算HouseHolder向量
        // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1));
        // 3、u=u/sqrt(abs(u(1)))
        T nu = 0.0;
        if (j == cols % blockDim.y) {
            // 0.求normx
            // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
            for (long k = 0; k < rowDataNum; k++) {
                acc[k] = 0.0;
                // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols) {
                    acc[k] = AA[i + k * blockDim.x + cols * ldaa] *
                             AA[i + k * blockDim.x + cols * ldaa];
                }
                nu += acc[k];
            }

            // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
            T norm_x_squre = warpAllReduceSum_ori(nu);
            T norm_x = sqrt(norm_x_squre);

            // if(blockIdx.x == 0 && i == 0 && j == 0) {
            //     printf("norm_x_squre: %f ", norm_x_squre);
            //     printf("\n");
            // }

            // 1、求u=x/norm(x);
            T scale = 1.0 / norm_x;
#pragma unroll
            for (long k = 0; k < rowDataNum; k++) {
                if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols) {
                    AA[i + k * blockDim.x + cols * ldaa] *= scale;
                }
            }

            __syncwarp();

            // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程来计算即可
            if (0 == i) {
                T u1 = AA[cols + cols * mm];

                AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;

                // 把normx存放到RR中，也就是对角线的元素
                RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
            }

            __syncwarp();

            // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
            scale = 1 / (sqrt(abs(AA[cols + cols * ldaa])));
#pragma unroll
            for (long k = 0; k < rowDataNum; k++) {
                if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols) {
                    AA[i + k * blockDim.x + cols * ldaa] *= scale;
                }
            }
        }

        __syncthreads();
        // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
        // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
        // 每个线程按列需要处理多个列
        for (int h = 0; h < colDataNum; h++) {
            T nu = 0.0;
            long opCols = j + h * blockDim.y;

            // 只更新当前列后面的列
            if (cols < opCols && opCols <= nn) {
                // 先计算u'x
#pragma unroll
                for (long k = 0; k < rowDataNum; k++) {
                    acc[k] = 0.0;
                    // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                    if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols) {
                        acc[k] = AA[i + k * blockDim.x + cols * ldaa] *
                                 AA[i + k * blockDim.x + opCols * ldaa];
                    }
                    nu += acc[k];
                }
                T utx = warpAllReduceSum_ori(nu);

                // 计算x-uu'x
#pragma unroll
                for (long k = 0; k < rowDataNum; k++) {
                    // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                    if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols) {
                        AA[i + k * blockDim.x + opCols * ldaa] -=
                            utx * AA[i + k * blockDim.x + cols * ldaa];
                    }
                }
                __syncwarp();
            }
        }
    }

    __syncthreads();
    // 此时已经完成HouseHolder更新，在AA中存放着HouseHolder向量和R矩阵的上三角部分,RR中存放在对角线元素

    // 获得R矩阵，将AA的上三角部分拷贝到R中
    // 以R矩阵来进行循环
    long rRowDataNum = (nn + (blockDim.x - 1)) / blockDim.x;
    for (int h = 0; h < colDataNum; h++) {
        long opCols = j + h * blockDim.y;

        if (opCols >= nn) continue;

#pragma unroll
        for (long k = 0; k < rRowDataNum; k++) {
            if (i + k * blockDim.x < opCols) {
                R[i + k * blockDim.x + opCols * ldr] =
                    AA[i + k * blockDim.x + opCols * ldaa];
                AA[i + k * blockDim.x + opCols * ldaa] = 0.0;
            } else if (i + k * blockDim.x > opCols) {
                R[i + k * blockDim.x + opCols * ldr] = 0.0;
            } else {
                // 这个赋值完全可以放到上面RR的赋值哪儿，从而不需要RR的共享内存
                R[opCols + opCols * ldr] = RR[opCols];
            }
        }
    }

    // if(blockIdx.x == 7 && i == 0 && j == 0) {
    //     printf("save to R\n");
    //     for(int v = 0; v < 32; v++) {
    //         for(int l = 0; l < 32; l++) {
    //             printf("%9.6f ", R[v + l * ldr]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // 来求Q，使用的方法是Q=(I-uu')Q, 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q
    // q表示是Q矩阵的1列
    // double q[rowDataNum * 2];
    T q[8 * 2];
    for (int h = 0; h < colDataNum; h++) {
        // 1、构造出每个线程需要处理的Q矩阵的一列q的一部分
        long opCols = j + h * blockDim.y;

        if (opCols >= nn) continue;

        for (long k = 0; k < rowDataNum; k++) {
            if (i + k * blockDim.x == opCols) {
                q[k] = 1.0;
            } else {
                q[k] = 0.0;
            }
        }

        __syncwarp();

        for (int cols = nn - 1; cols >= 0; cols--) {
            // 这个判断没有问题，很经典，实际上不带这个判断也是正确的。这个判断是利用矩阵特点对矩阵乘法的一种优化
            // 因为Q_k-1=(I-u_k-1*u_k-1')*Q_k-2也是一个左上角是单位矩阵，右下角是一个k-1xk-1的矩阵，其他部分都是0；
            // 而I-uk*uk'也是一个左上角是单位矩阵，右下角是一个kxk的矩阵，其他部分为0；所以两者相乘只影响后面大于等于k的列
            if (opCols >= cols) {
                // 2、计算u'q
                T nu = 0.0;
                for (long k = 0; k < rowDataNum; k++) {
                    acc[k] = 0.0;
                    if (i + k * blockDim.x < mm) {
                        acc[k] = AA[i + k * blockDim.x + cols * ldaa] * q[k];
                    }
                    nu += acc[k];
                }

                T utq = warpAllReduceSum_ori(nu);

                // 3.计算q-uu'q
                for (long k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim.x < mm) {
                        q[k] -= utq * AA[i + k * blockDim.x + cols * ldaa];
                    }
                }

                __syncwarp();
            }
        }

        // 4.把计算出来的q拷贝到A中
        for (long k = 0; k < rowDataNum; k++) {
            if (i + k * blockDim.x < mm) {
                A[i + k * blockDim.x + opCols * lda] = q[k];
            }
        }
    }

    __syncthreads();
    // if(blockIdx.x == 0 && i == 0 && j == 0) {
    //     printf("save to A\n");
    //     for(int v = 0; v < mm; v++) {
    //         for(int l = 0; l < 32; l++) {
    //             printf("%9.6f ", A[v + l * lda]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}

template __global__ void my_hou_kernel_ori<float, 128, 32>(long m, long n, float *A, long lda, float *R, long ldr);
template __global__ void my_hou_kernel_ori<double, 128, 32>(long m, long n, double *A, long lda, double *R, long ldr);

