#include <stdio.h>
#include <stdlib.h>

#define CSC(call)                    \
do {                                \
    cudaError_t res = call;            \
    if (res != cudaSuccess) {        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                    \
    }                                \
} while(0)

__global__ void kernel(double *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (idx < n / 2) {
        double tmp = arr[idx];
        arr[idx] = arr[n - idx - 1];
        arr[n - idx - 1] = tmp;

        idx += offset;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    double *input_arr = (double *) malloc(sizeof(double) * n);
    if (input_arr == NULL) {
        printf("ERROR: in %s:%d. Not enough memory.\n", __FILE__, __LINE__);
        return 0;
    }
    for (int i = 0; i < n; i++)
        scanf("%lf", &input_arr[i]);

    double *device_arr;
    CSC(cudaMalloc(&device_arr, sizeof(double) * n));
    CSC(cudaMemcpy(device_arr, input_arr, sizeof(double) * n, cudaMemcpyHostToDevice));

    kernel << < 256, 256 >> > (device_arr, n);
    CSC(cudaGetLastError());


    CSC(cudaMemcpy(input_arr, device_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(device_arr));

    for (int i = 0; i < n; i++)
        printf("%.10e ", input_arr[i]);
    printf("\n");
    free(input_arr);
    return 0;
}
