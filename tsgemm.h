template <typename T>
__global__ void tsgemm(int m, int n, int k, T alpha, int trans_A, T *A,
                       const int lda, T *B, const int ldb, T beta, T *C,
                       const int ldc) {
    const int grid_dim_x = gridDim.x, grid_dim_y = gridDim.y;
    const int block_dim_x = BLOCK_DIM_X, block_dim_y = BLOCK_DIM_Y;
    const int block_idx_x = blockIdx.x, block_idx_y = blockIdx.y;
    const int thread_idx_x = threadIdx.x, thread_idx_y = threadIdx.y;

    const int num_row =
        (m + grid_dim_x * block_dim_x - 1) / grid_dim_x * block_dim_x;
    const int num_col =
        (n + grid_dim_y * block_dim_y - 1) / grid_dim_y * block_dim_y;

    T c_per_thread[NUM_Q_ROW * NUM_Q_COL];

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
                c_per_thread[row_repeat_idx + col_repeat_idx * NUM_Q_ROW] = sum;
            }
        }
    } else {  // trans_A == 1
        for (int col_repeat_idx = 0; col_repeat_idx < num_col;
             ++col_repeat_idx) {
            int col_idx = col_repeat_idx * grid_dim_y * block_dim_y +
                          block_idx_y * block_dim_y + thread_idx_y;
            if (col_idx >= n) break;
            for (int row_repeat_idx = 0; row_repeat_idx < num_row;
                 ++row_repeat_idx) {
                int row_idx = row_repeat_idx * grid_dim_x * block_dim_x +
                              block_idx_x * block_dim_x + thread_idx_x;
                if (row_idx >= m) break;
                T sum = 0;
                for (int i = 0; i < k; ++i) {
                    sum += A[row_idx * lda + i] * B[i + col_idx * ldb];
                }
                c_per_thread[row_repeat_idx + col_repeat_idx * NUM_Q_ROW] = sum;
            }
        }
    }

    for (int row_repeat_idx = 0; row_repeat_idx < num_row; ++row_repeat_idx) {
        int row_idx = row_repeat_idx * grid_dim_x * block_dim_x +
                      block_idx_x * block_dim_x + thread_idx_x;
        if (row_idx >= m) break;
        for (int col_repeat_idx = 0; col_repeat_idx < num_col;
             ++col_repeat_idx) {
            int col_idx = col_repeat_idx * grid_dim_y * block_dim_y +
                          block_idx_y * block_dim_y + thread_idx_y;
            if (col_idx >= n) break;
            C[row_idx + col_idx * ldc] =
                alpha *
                    c_per_thread[row_repeat_idx + col_repeat_idx * NUM_Q_ROW] +
                beta * C[row_idx + col_idx * ldc];
        }
    }
}
template __global__ void tsgemm<double>(int m, int n, int k, double alpha,
                                        int trans_A, double *A, const int lda,
                                        double *B, const int ldb, double beta,
                                        double *C, const int ldc);
template __global__ void tsgemm<float>(int m, int n, int k, float alpha,
                                       int trans_A, float *A, const int lda,
                                       float *B, const int ldb, float beta,
                                       float *C, const int ldc);
