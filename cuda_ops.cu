#include <iostream>
#include <cuda_runtime.h>
using namespace std;

const int N = 512;

__global__ void matAdd(int* a, int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n)
        c[row * n + col] = a[row * n + col] + b[row * n + col];
}

__global__ void matMul(int* a, int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++)
            sum += a[row * n + k] * b[k * n + col];
        c[row * n + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);

    int *a = new int[N*N], *b = new int[N*N];
    int *add = new int[N*N], *mul = new int[N*N];

    for (int i = 0; i < N * N; i++) { a[i] = 1; b[i] = 1; }

    int *da, *db, *dadd, *dmul;
    cudaMalloc(&da,   size);
    cudaMalloc(&db,   size);
    cudaMalloc(&dadd, size);
    cudaMalloc(&dmul, size);

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    matAdd<<<blocks, threads>>>(da, db, dadd, N);
    cudaMemcpy(add, dadd, size, cudaMemcpyDeviceToHost);

    matMul<<<blocks, threads>>>(da, db, dmul, N);
    cudaMemcpy(mul, dmul, size, cudaMemcpyDeviceToHost);

    cout << "Matrix size: " << N << "x" << N << "\n\n";

    cout << "== Matrix Addition ==\n";
    cout << "C[0][0] = " << add[0]   << " (expected 2)\n";
    cout << "C[1][1] = " << add[N+1] << " (expected 2)\n\n";

    cout << "== Matrix Multiplication ==\n";
    cout << "C[0][0] = " << mul[0]   << " (expected " << N << ")\n";
    cout << "C[1][1] = " << mul[N+1] << " (expected " << N << ")\n";

    cudaFree(da); cudaFree(db); cudaFree(dadd); cudaFree(dmul);
    delete[] a; delete[] b; delete[] add; delete[] mul;
    return 0;
}
