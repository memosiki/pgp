#include <numeric>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <string.h>

#define MAX_N 1000

#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)
__constant__ double cur_row[MAX_N];

__global__ void kernel(double* dev_a, int i, int n){
    int start_x = gridDim.x * blockIdx.x + threadIdx.x;
    int start_y = gridDim.y * blockIdx.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;

    double ratio;
    for (int y = i + 1 + start_y; y < n; y += offset_y) {
        ratio = dev_a[y * MAX_N + i] / cur_row[i];
        for (int x = i + 1 + start_x; x < n; x += offset_x){
            dev_a[y * MAX_N + x] -= ratio * cur_row[x];
            // printf("upd %d %d\n",x, y );
        }
    }
}
double a[MAX_N][MAX_N];
int cur_i;
bool cmp(int el1, int el2){
    // компаратор -- сравнивает значения в столбцах у строк по заданным значениям
    // номер столбца задаётся глобально
    return abs(a[el1][cur_i]) < abs(a[el2][cur_i]);
}
int main() {
     // double a[10000][10000];
    int n;
    double eps = 1e-7;
    scanf("%d", &n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            scanf("%lf", &a[i][j]);
            // храним по строкам
    double *dev_a;
    CSC(cudaMalloc( (void**) &dev_a, sizeof(double) * MAX_N * MAX_N));
    int pos[MAX_N];
    std::iota(pos, pos+n, 0);
    int sign = 1;
    double det = 0;
    for (int i = 0; i < n; ++i) {
        // выбираем максимальный
        cur_i = i;
        int* it;
        it = std::max_element(pos+i, pos+n, cmp);
        int major = it - pos;


        // printf("%d %d %lf\n", i, major, a[major][i]);
        if (i == n-1){
            // если последняя строка
            if (abs(a[i][i]) < eps ){
                printf("det %.10e\n", 0.);
                return 0;
            }
            else if (a[i][i] < 0){
                det += log( -a[i][i] );
                sign = - sign;
            } else
                det += log( a[i][i] );
            break;
        }
        CSC(cudaMemcpyToSymbol(cur_row, a[major], sizeof(double) * n, 0, cudaMemcpyHostToDevice));

        if (abs(a[major][i]) < eps ){
            printf("det %.10e\n", 0.);
            return 0;
        } else if (a[major][i] < 0){
            det += log( -a[major][i] );
            sign = - sign;
        } else
            det += log( a[major][i] );

        if (major != i){
            memcpy((double*)a + major * MAX_N,  (double*)a + i * MAX_N, MAX_N * sizeof(double));
            sign = -sign;
        }
        // CSC(cudaMemcpy(dev_a + i * MAX_N, a + i * MAX_N, sizeof(double) * MAX_N * (MAX_N - i), cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_a, a, sizeof(double) * MAX_N * MAX_N, cudaMemcpyHostToDevice));
        // printf("aaaaa\n");

        //вычитаем из каждой строчки текущую
        kernel <<< dim3(32, 32), dim3(32, 32) >>> (dev_a, i, n);

        CSC(cudaGetLastError());
        // CSC(cudaMemcpy((double*)a + i * MAX_N, (double*) dev_a + i * MAX_N, sizeof(double) * MAX_N * (MAX_N - i), cudaMemcpyDeviceToHost));
        CSC(cudaMemcpy(a, dev_a, sizeof(double) * MAX_N * MAX_N, cudaMemcpyDeviceToHost));

    }
    double ans = sign*exp(det);
    printf("det %.10e\n", ans);
    return 0;
}
