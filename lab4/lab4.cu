#include <numeric>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <string.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define MAX_ROW 10000
#define MAX_COL 16384
// __device__ double a_GPU[MAX_ROW*MAX_N];

#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)

// __constant__ double cur_column[MAX_COL];

__global__ void swapRow(double* dev_a, int y1, int y2, int n){
    int idx = gridDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    double buffer;
    while(idx < n){
        buffer = dev_a[idx*MAX_COL + y1];
        dev_a[idx*MAX_COL + y1] = dev_a[idx*MAX_COL + y2];
        dev_a[idx*MAX_COL + y2] = buffer;
        idx += offset;
    }
}
__global__ void print_mat(double* dev_a, int n){
    int start_x = gridDim.x * blockIdx.x + threadIdx.x;
    int start_y = gridDim.y * blockIdx.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;
    for (int x = start_x; x < n; x += offset_x)
        for (int y = start_y; y < n; y += offset_y)
            printf("%d %d %lf\n", x, y, dev_a[x * MAX_COL + y]);
}
__global__ void kernel(double* dev_a, int i, int n, double major_elem){
    int start_x = gridDim.x * blockIdx.x + threadIdx.x;
    int start_y = gridDim.y * blockIdx.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;
    // for (int y = start_y; y < n; y += offset_y)
    //     for (int x = start_x; x < n; x += offset_x)
    //         printf("%d %d %lf\n", x, y, dev_a[x * MAX_COL + y]);
    double ratio;
    for (int y = i + 1 + start_y; y < n; y += offset_y){
        // printf("%d %lf from %lf / %lf\n", y, ratio, dev_a[i * MAX_COL + y], major_elem);
        ratio = dev_a[i * MAX_COL + y] / major_elem;
        for (int x = i + 1 + start_x; x < n; x += offset_x){
            // printf("rat %lf  = %lf / major at %d %d\n", ratio, dev_a[i * MAX_COL + y], i, y);
            dev_a[x * MAX_COL + y] -= ratio * dev_a[x * MAX_COL + i];
            // printf("changing %d %d new %lf  -= %lf %lf\n", x, y, dev_a[x * MAX_COL + y], ratio, dev_a[x * MAX_COL + i]);
            // printf("minus %lf from %d %d rat %lf\n", ratio * dev_a[x * MAX_COL + i], x, y, ratio);
        }
    }
}
double a[MAX_ROW * MAX_COL];

struct comparator {
	__host__ __device__ bool operator()(double a, double b) {
		return abs(a) < abs(b);
	}
};
__host__ int align_n(int n){
        // выравнивает n, чтобы оно делилось на 32 нацело
        if (n % 32)
            return n + (32 - n % 32);
        return n;
}
int main(int argc, char const *argv[]) {
    FILE* inp = stdin;
    if (argc > 1)
        inp = fopen(argv[1], "r");
    // inp = fopen("config3.data", "r");
    int n;
    double eps = 1e-7;
    fscanf(inp, "%d", &n);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            fscanf(inp, "%lf", &a[x*MAX_COL + y]);
            // храним по столбцам
    // for (int x = 0; x < n; ++x)
    //     for (int y = 0; y < n; ++y)
    //         printf("%lf ", a[x][y]);

    double *dev_a;
    // CSC(cudaGetSymbolAddress((void **)&dev_a, a_GPU));
    CSC(cudaMalloc( &dev_a, sizeof(double) * MAX_COL * n));
    CSC(cudaMemcpy(dev_a, a, sizeof(double) * MAX_COL * n, cudaMemcpyHostToDevice));

    int sign = 1;
    double det = 0;
    int major;
    double major_elem;
    // double pos[10];
    // std::iota(pos, pos + 10, 0);
    // std::reverse(pos, pos+10);
    // CSC(cudaMalloc(&dev_a, sizeof(double) * 10));
    // CSC(cudaMemcpy(dev_a, pos, sizeof(double) * 10, cudaMemcpyHostToDevice));
    thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(dev_a);
    comparator comp;
    int aligned_n = n;//align_n(n);
    thrust::device_ptr<double> res;
    for (int i = 0; i < n; ++i) {
        try {
                res = thrust::max_element(
                    p_arr + i * MAX_COL + i,
                    p_arr + i * MAX_COL + n,
                    comp
                );
        } catch(thrust::system_error e) {
          std::cerr << "Error inside sort: " << e.what() << std::endl;
        }
        major_elem = *res;
        major = (res - p_arr) % MAX_COL;
        // printf("////////////");
        // printf("%d %lf at %d\n", i, major_elem, major);
        if (major != i){
            swapRow <<< 256, 256>>>(dev_a, i, major, aligned_n);
            sign = -sign;
        }

        if (abs(major_elem) < eps ){
            printf("det %.10e\n", 0.);
            return 0;
        } else if (major_elem < 0){
            det += log( -major_elem );
            sign = - sign;
        } else
            det += log( major_elem );
        if (i == n-1)
            break;
        // print_mat <<< dim3(32, 32), dim3(32, 32) >>> (dev_a, n);

        kernel <<< dim3(32, 32), dim3(32, 32) >>> (dev_a, i, aligned_n, major_elem);
        // major = i;
        // CSC(cudaMemcpyToSymbol(cur_row, dev_a + major * MAX_ROW , sizeof(double) * n, 0, cudaMemcpyDeviceToDevice));
        // CSC(cudaMemcpy(&major_elem, dev_a + major * MAX_ROW + i , sizeof(double), cudaMemcpyDeviceToHost));
        // a[major][i] = major_elem;
        // // printf("%d %lf\n", i, a[major][i]);

        //
        //
        // //вычитаем из каждой строчки текущую
        // kernel <<< dim3(32, 32), dim3(32, 32) >>> (dev_a, i, n);
        // CSC(cudaGetLastError());

    }
    double ans = sign*exp(det);
    printf("det %.10e\n", ans);
    return 0;
}
