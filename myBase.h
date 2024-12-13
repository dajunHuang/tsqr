#pragma once

#define CHECK(call)                                                         \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// #define MY_DEBUG 1

// 下面的部分只在单个文件的内部进行调用
static cudaEvent_t start, stop;
static void startTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

static float stopTimer() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

template <typename T>
void printDeviceMatrixV2(T* dA, long ldA, long rows, long cols) {
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

template void printDeviceMatrixV2(double* dA, long ldA, long rows, long cols);
template void printDeviceMatrixV2(float* dA, long ldA, long rows, long cols);

template <typename T>
bool all_close(T const* A, T const* A_ref, size_t m, size_t n, size_t lda,
               T abs_tol, double rel_tol) {
    bool status{true};
    for (size_t j{0U}; j < n; ++j) {
        for (size_t i{0U}; i < m; ++i) {
            double const A_val{static_cast<double>(A[i + j * lda])};
            double const A_ref_val{static_cast<double>(A_ref[i + j * lda])};
            double const diff{A_val - A_ref_val};
            double const diff_val{std::abs(diff)};
            if (std::isnan(diff_val) || diff_val >
                std::max(static_cast<double>(abs_tol),
                         static_cast<double>(std::abs(A_ref_val)) * rel_tol)) {
                std::cout << "A[" << i << ", " << j << "] = " << A_val
                          << " A_ref[" << i << ", " << j << "] = " << A_ref_val
                          << " Abs Diff: " << diff_val
                          << " Abs Diff Threshold: "
                          << static_cast<double>(abs_tol)
                          << " Rel->Abs Diff Threshold: "
                          << static_cast<double>(
                                 static_cast<double>(std::abs(A_ref_val)) *
                                 rel_tol)
                          << std::endl;
                status = false;
                return status;
            }
        }
    }
    return status;
}

template bool all_close(double const* A, double const* A_ref, size_t m,
                        size_t n, size_t lda, double abs_tol, double rel_tol);
