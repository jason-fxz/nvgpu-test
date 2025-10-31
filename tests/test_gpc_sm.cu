#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#define CHECKRT(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
const size_t N = 40 * 1024 * 1024; // 40 MB
int total_sm_count = 108;
int *d_data;
unsigned long long *d_time;
__device__ __forceinline__ unsigned int get_smid(void) {
    unsigned int ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}
__device__ __forceinline__ bool is_smid_active(uint64_t smid_mask_lower, uint64_t smid_mask_upper, unsigned smid) {
    if (smid < 64) {
        return (smid_mask_lower >> smid) & 1;
    } else {
        return (smid_mask_upper >> (smid - 64)) & 1;
    }
}
__global__ void l2_bandwidth_kernel(uint64_t smid_mask_lower,
                                    uint64_t smid_mask_upper,
                                    int *data,
                                    unsigned long long *out_time,
                                    size_t N, int iters)
{
    unsigned smid = get_smid();
    // 如果不是指定的 SMID, 直接退出，标记一下 -1
    if (!(is_smid_active(smid_mask_lower, smid_mask_upper, smid))) {
        // just return
        if (threadIdx.x == 0) {
            out_time[smid] = -1ull;
        }
        return ;
    }
    size_t elems_per_sm = N / 128;
    int *data_base = data + smid * elems_per_sm;
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    // __syncthreads();
    unsigned long long start = clock();
    volatile int *vptr = data_base;
    int val = (int)tid;
    for (int t = 0; t < iters; t++) {
        #pragma unroll 32
        for (size_t i = tid; i < elems_per_sm; i += stride) {
            // 直接写全局内存，绕过 L1：st.global.cg
            asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(vptr + i), "r"(val + t));
        }
    }
    // __syncthreads();
    unsigned long long end = clock();
    if (threadIdx.x == 0) {
        out_time[smid] = end - start;
    }
}
bool check_mask(uint64_t mask_lower, uint64_t mask_upper, int i) {
    if (i < 64) {
        return (mask_lower >> i) & 1;
    } else {
        return (mask_upper >> (i - 64)) & 1;
    }
}
void test_on_sm(uint64_t mask_lower, uint64_t mask_upper) {
    cudaEvent_t start, stop;
    CHECKRT(cudaEventCreate(&start));
    CHECKRT(cudaEventCreate(&stop));
    int blocks = 108;
    int threads = 256;
    CHECKRT(cudaMemset(d_time, 0, total_sm_count * sizeof(unsigned long long)));
    cudaDeviceSynchronize();
    CHECKRT(cudaEventRecord(start));
    // launch kernel
    l2_bandwidth_kernel<<<blocks, threads>>>(mask_lower, mask_upper, d_data, d_time, N, 1000);
    CHECKRT(cudaEventRecord(stop));
    CHECKRT(cudaEventSynchronize(stop));
    unsigned long long h_time[128];
    CHECKRT(cudaMemcpy(h_time, d_time, total_sm_count * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    printf("SM: ");
    for (int i = 0; i < total_sm_count; ++i) {
        if (check_mask(mask_lower, mask_upper, i)) {
            // printf("%d ", i);
            // if (h_time[i] > tmax) tmax = h_time[i];
            if (h_time[i] == 0) {
                printf("\nSM %d is inactive\n", i);
            }
        } else {
            if (h_time[i] != -1Ull) {
                printf("\nWTF SM %d with time %llu\n", i, h_time[i]);
            }
        }
    }
    for (int i = 0; i < total_sm_count; ++i) {
        if (check_mask(mask_lower, mask_upper, i)) {
            printf("%d:%llu ", i, h_time[i]);
        }
    }
    printf("\n");
    // printf("    | Time %f\n", milliseconds);
    CHECKRT(cudaEventDestroy(start));
    CHECKRT(cudaEventDestroy(stop));
}
int main() {
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_time, total_sm_count * sizeof(unsigned long long));
    // warmup
    test_on_sm(~0ULL, ~0ULL);
    test_on_sm(~0ULL, ~0ULL);
    test_on_sm(~0ULL, ~0ULL);
    test_on_sm(~0ULL, ~0ULL);
    test_on_sm(~0ULL, ~0ULL);
    test_on_sm(~0ULL, ~0ULL);
    test_on_sm(0b00000000'00000001ull<<28, 0);
    test_on_sm(0b00000000'00000101ull<<28, 0);
    test_on_sm(0b00000000'00010101ull<<28, 0);
    test_on_sm(0b00000000'01010101ull<<28, 0);
    test_on_sm(0b00000001'01010101ull<<28, 0);
    test_on_sm(0b00000101'01010101ull<<28, 0);
    test_on_sm(0b00010101'01010101ull<<28, 0);
    test_on_sm(0b01010101'01010101ull<<28, 0);
    test_on_sm((1ull << 0), 0);
    test_on_sm((1ull << 0) | (1ull << 7), 0);
    test_on_sm((1ull << 0) | (1ull << 7) | (1ull << 14), 0);
    test_on_sm((1ull << 0) | (1ull << 7) | (1ull << 14) | (1ull << 21), 0);
    test_on_sm((1ull << 0) | (1ull << 7) | (1ull << 14) | (1ull << 21) | (1ull << 28), 0);
    test_on_sm((1ull << 0) | (1ull << 7) | (1ull << 14) | (1ull << 21) | (1ull << 28) | (1ull << 35), 0);
    test_on_sm((1ull << 0) | (1ull << 7) | (1ull << 14) | (1ull << 21) | (1ull << 28) | (1ull << 35) | (1ull << 42), 0);
    cudaFree(d_data);
    cudaFree(d_time);
    return 0;
}