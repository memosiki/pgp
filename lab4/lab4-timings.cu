#include <stdio.h>
#include <math.h>
#include <string.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define MAX_ROW 10000
#define MAX_COL 10048
#define BLOCK 16

#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)


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

__global__ void kernel(double* dev_a, int i, int aligned_i, int n, int aligned_n, double major_elem){
    int start_row = gridDim.x * blockIdx.x + threadIdx.x;
    int start_col = blockIdx.y;

    int offset_row = gridDim.x * blockDim.x;
    int offset_col = gridDim.y;

    double ratio;
    for (int x = i + 1 + start_col; x < n; x += offset_col){
        __syncthreads();
        // broadcasted
        ratio = dev_a[x * MAX_COL + i] / major_elem;
        for (int y = aligned_i + start_row; y < aligned_n; y += offset_row){
            if ( i<x && x<n && i<y && y<n){
                __syncthreads();
                // aligned in memory --> get/store coalescing
                dev_a[x * MAX_COL + y] -= ratio * dev_a[i * MAX_COL + y];
            }
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
    // aligns n by BLOCK value
    if (n % BLOCK)
        return n + (BLOCK - n % BLOCK);
    return n;
}

__host__ int align_down(int n){
    // aligns n to ceil by BLOCK value
    if (n % BLOCK)
        return n - n % BLOCK;
    return n;
}

int main(int argc, char const *argv[]) {
    FILE* inp = stdin;
    if (argc > 1)
        inp = fopen(argv[1], "r");
    int n;
    double eps = 1e-7;
    fscanf(inp, "%d", &n);
    if (n == 0){
        printf("%.10e\n", 0.);
        return 0;
    }

    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            fscanf(inp, "%lf", &a[x*MAX_COL + y]);
            // храним по столбцам

    double *dev_a;
    size_t pitch;
    CSC(cudaMallocPitch( &dev_a, &pitch, sizeof(double) * MAX_COL, n));
    // check that its already aligned by strategically chosen MAX_COL
    assert(pitch/sizeof(double) == MAX_COL);
    CSC(cudaMemcpy(dev_a, a, sizeof(double) * MAX_COL * n, cudaMemcpyHostToDevice));

    int sign = 1;
    double det = 0;
    int major;
    double major_elem;

    thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(dev_a);
    comparator comp;
    thrust::device_ptr<double> res;

    int aligned_n = align_n(n);
    int aligned_col_count, aligned_row_count, aligned_i;


    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    for (int i = 0; i < n; ++i) {
        // try {
        res = thrust::max_element(
            p_arr + i * MAX_COL + i,
            p_arr + i * MAX_COL + n,
            comp
        );
        // } catch(thrust::system_error e) {
        //   std::cerr << "Error inside sort: " << e.what() << std::endl;
        // }
        major_elem = *res;
        major = (res - p_arr) % MAX_COL;
        if (major != i){
            swapRow <<< 256, 256>>>(dev_a, i, major, n);
            sign = -sign;
        }

        if (abs(major_elem) < eps ){
            printf("%.10e\n", 0.);
            return 0;
        } else if (major_elem < 0){
            det += log( -major_elem );
            sign = - sign;
        } else
            det += log( major_elem );
        if (i == n-1)
            break;

        aligned_i = align_down(i);
        aligned_col_count = ceil((float) (n-aligned_i) / BLOCK);
        aligned_row_count = min(n-aligned_i , BLOCK);

        kernel <<< dim3(aligned_row_count, aligned_col_count), BLOCK >>>
                (dev_a, i, aligned_i, n, aligned_n, major_elem);

        CSC(cudaGetLastError());

    }
    double ans = sign*exp(det);


    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	fprintf(stderr, "time = %f\n", time);
	cudaEventDestroy(stop);
	cudaEventDestroy(start);

    printf("%.10e\n", ans);
    return 0;
}
