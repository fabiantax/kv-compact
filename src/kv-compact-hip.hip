// KV Cache Compaction — ROCm/HIP Accelerated Kernels
//
// GPU-accelerated matrix multiply for the attention scoring hotspot in
// Step 1 (Section 3.1) of the compaction algorithm from:
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
// The dominant cost is computing Q_ref @ K^T / sqrt(d_k) for all heads,
// which is an (n_q × d_k) * (T × d_k)^T matmul. On CPU this is O(n_q * T * d_k)
// per head per layer; GPU acceleration helps when T is large (>1K tokens).
//
// Target: AMD Ryzen AI 395 Pro Max with integrated RDNA 3.5 GPU (8060S)
// The integrated GPU shares system memory, so no explicit D↔H copies needed
// for unified memory allocations (hipMallocManaged).

#include <hip/hip_runtime.h>
#include "kv-compact-accel.h"

#include <cstdio>
#include <cstdlib>

// ============================================================================
// HIP Kernels
// ============================================================================

// Matrix multiply: C = A * B^T
// A is (m x k), B is (n x k), C is (m x n)
// Each thread computes one element of C
__global__ void kernel_mat_mul_ABt(
        const float * __restrict__ A,
        const float * __restrict__ B,
        float * __restrict__ C,
        int m, int n, int k, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[col * k + l];
        }
        C[row * n + col] = sum * scale;
    }
}

// Tiled matrix multiply for better cache utilization
// Uses shared memory tiles for both A and B
#define TILE_SIZE 16

__global__ void kernel_mat_mul_ABt_tiled(
        const float * __restrict__ A,
        const float * __restrict__ B,
        float * __restrict__ C,
        int m, int n, int k, float scale) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_col = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? A[row * k + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (col < n && b_col < k) ? B[col * k + b_col] : 0.0f;

        __syncthreads();

        for (int l = 0; l < TILE_SIZE; l++) {
            sum += tile_A[threadIdx.y][l] * tile_B[l][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum * scale;
    }
}

// ============================================================================
// Public API (C-linkage)
// ============================================================================

static bool hip_initialized = false;
static bool hip_available = false;

static void ensure_hip_init() {
    if (hip_initialized) return;
    hip_initialized = true;

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        hip_available = false;
        return;
    }

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    fprintf(stderr, "[kv-compact] HIP device: %s (CUs: %d, clock: %d MHz)\n",
            props.name, props.multiProcessorCount, props.clockRate / 1000);

    hip_available = true;
}

extern "C" {

int kv_compact_hip_available(void) {
    ensure_hip_init();
    return hip_available ? 1 : 0;
}

int kv_compact_hip_mat_mul_ABt_scaled(
        const float * A, const float * B, float * C,
        int m, int n, int k, float scale) {
    ensure_hip_init();
    if (!hip_available) return -1;

    float *d_A, *d_B, *d_C;
    size_t size_A = (size_t)m * k * sizeof(float);
    size_t size_B = (size_t)n * k * sizeof(float);
    size_t size_C = (size_t)m * n * sizeof(float);

    // Use hipMallocManaged for unified memory (APU optimization —
    // avoids explicit copies on integrated GPU with shared memory)
    hipMallocManaged(&d_A, size_A);
    hipMallocManaged(&d_B, size_B);
    hipMallocManaged(&d_C, size_C);

    memcpy(d_A, A, size_A);
    memcpy(d_B, B, size_B);

    // Use tiled kernel for larger problems, simple kernel for small ones
    if (m >= TILE_SIZE && n >= TILE_SIZE) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
                  (m + TILE_SIZE - 1) / TILE_SIZE);
        kernel_mat_mul_ABt_tiled<<<grid, block>>>(d_A, d_B, d_C, m, n, k, scale);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (m + 15) / 16);
        kernel_mat_mul_ABt<<<grid, block>>>(d_A, d_B, d_C, m, n, k, scale);
    }

    hipDeviceSynchronize();
    memcpy(C, d_C, size_C);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}

} // extern "C"
