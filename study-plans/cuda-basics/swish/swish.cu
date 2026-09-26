#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    // Hər bir thread-in qlobal indeksini təyin edirik
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Əgər indeks N-dən kiçikdirsə, hesablamanı aparırıq (Guard)
    if (i < N) {
        float x = input[i];
        // Sigmoid funksiyası: 1 / (1 + e^(-x))
        // Sürət üçün __expf istifadə edirik
        float sigmoid = 1.0f / (1.0f + __expf(-x));
        
        // Swish = x * sigmoid(x)
        output[i] = x * sigmoid;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    // Blok sayını tapırıq (yuvarlaqlama yuxarı - ceil division)
    int blocks = (N + threads - 1) / threads;
    
    // Kerneli işə salırıq
    swish_kernel<<<blocks, threads>>>(input, output, N);
    
    // GPU-nun işini bitirməsini gözləyirik
    cudaDeviceSynchronize();
}