#include <stdio.h>
#include <algorithm>
#include <float.h>
#include <stdint.h>

#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)


typedef float TMatrix[3][3];
typedef float3 TVector;
const int MAX_CLASSES = 32;

__constant__ TVector
avgGPU[MAX_CLASSES];
__constant__ TMatrix
covGPU[MAX_CLASSES];

__device__ __host__

void copy_matrix33(TMatrix to, TMatrix from) {
    to[0][0] = from[0][0];
    to[0][1] = from[0][1];
    to[0][2] = from[0][2];
    to[1][0] = from[1][0];
    to[1][1] = from[1][1];
    to[1][2] = from[1][2];
    to[2][0] = from[2][0];
    to[2][1] = from[2][1];
    to[2][2] = from[2][2];
}


__host__ float determ33(TMatrix a) {
    return a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) -
           a[0][1] * (a[1][0] * a[2][2] - a[2][0] * a[1][2]) +
           a[0][2] * (a[1][0] * a[2][1] - a[2][0] * a[1][1]);
}

__host__ void inverse33(TMatrix a) {
    float det = determ33(a);
    TMatrix tmp;
    tmp[0][0] = 1 / det * (a[1][1] * a[2][2] - a[2][1] * a[1][2]);
    tmp[0][1] = -1 / det * (a[1][0] * a[2][2] - a[2][0] * a[1][2]);
    tmp[0][2] = 1 / det * (a[1][0] * a[2][1] - a[2][0] * a[1][1]);
    tmp[1][0] = -1 / det * (a[0][1] * a[2][2] - a[2][1] * a[0][2]);
    tmp[1][1] = 1 / det * (a[0][0] * a[2][2] - a[2][0] * a[0][2]);
    tmp[1][2] = -1 / det * (a[0][0] * a[2][1] - a[2][0] * a[0][1]);
    tmp[2][0] = 1 / det * (a[0][1] * a[1][2] - a[1][1] * a[0][2]);
    tmp[2][1] = -1 / det * (a[0][0] * a[1][2] - a[1][0] * a[0][2]);
    tmp[2][2] = 1 / det * (a[0][0] * a[1][1] - a[1][0] * a[0][1]);
    std::swap(tmp[0][1], tmp[1][0]);
    std::swap(tmp[0][2], tmp[2][0]);
    std::swap(tmp[1][2], tmp[2][1]);
    copy_matrix33(a, tmp);
}

__device__ float get_dist(uchar4 p, int j) {
    TMatrix cov; //inverse cov
    TVector mean = avgGPU[j];
    TVector diff = make_float3(0, 0, 0);
    TVector c; // for matmul

    copy_matrix33(cov, covGPU[j]);

    diff.x = p.x - mean.x;
    diff.y = p.y - mean.y;
    diff.z = p.z - mean.z;

    c.x = diff.x * cov[0][0] + diff.y * cov[1][0] + diff.z * cov[2][0];
    c.y = diff.x * cov[0][1] + diff.y * cov[1][1] + diff.z * cov[2][1];
    c.z = diff.x * cov[0][2] + diff.y * cov[1][2] + diff.z * cov[2][2];

    float dist = c.x * diff.x + c.y * diff.y + c.z * diff.z;
    return dist * (-1.0);
}

__global__ void kernel(uchar4 *dev_arr, size_t w, size_t h, int nc) {
    size_t idx = (size_t) threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = (size_t) blockDim.x * gridDim.x;
    float dist, new_dist; // Mahalanobis dist
    int cls = nc - 1; // class
    uchar4 p;

    while (idx < w * h) {
        p = dev_arr[idx];
        dist = -FLT_MAX;
        for (int j = 0; j < nc; ++j) {
            new_dist = get_dist(p, j);

            if (new_dist > dist) {
                dist = new_dist;
                cls = j;
            }
            if (new_dist == dist)
                if (cls > j)
                    cls = j;
        }
        dev_arr[idx].w = cls;
        idx += offset;
    }
}

int main() {
    char input_file[256];
    char output_file[256];
    size_t w, h;
    int nc, np;
    int x, y;

    uchar4 *data;
    float3 mean_arr[MAX_CLASSES];
    TMatrix cov_arr[MAX_CLASSES];
    uchar2 *pxls = (uchar2 *) malloc(sizeof(uchar4) * 2 * 524288 * 2); // 2**19=524288

    scanf("%s", input_file);
    FILE *file_in = fopen(input_file, "rb");
    uint w_int, h_int;
    fread(&w_int, sizeof(int), 1, file_in);
    fread(&h_int, sizeof(int), 1, file_in);
    w = w_int;
    h = h_int;

    data = (uchar4 *) malloc(h * w * sizeof(uchar4));
    fread(data, sizeof(uchar4), h * w, file_in);
    fclose(file_in);

    scanf("%s", output_file);
    scanf("%d", &nc);
    if (nc < 1)
        // incorrect number of classes
        return -1;
    for (int i = 0; i < nc; ++i) {
        scanf("%d", &np);
        if (np == 0) {
            mean_arr[i] = make_float3(0, 0, 0);
            TMatrix cov = {{0, 0, 0},
                           {0, 0, 0},
                           {0, 0, 0}};
            copy_matrix33(cov_arr[i], cov);
        }
        float3 mean = make_float3(0, 0, 0);
        for (int j = 0; j < np; ++j) {
            scanf("%d%d", &x, &y);
            pxls[j].x = x;
            pxls[j].y = y;
            uchar4 pj = data[w * y + x];
            mean.x += pj.x;
            mean.y += pj.y;
            mean.z += pj.z;
        }

        mean.x *= float(1) / np;
        mean.y *= float(1) / np;
        mean.z *= float(1) / np;
        mean_arr[i] = mean;
        TMatrix cov = {{0, 0, 0},
                       {0, 0, 0},
                       {0, 0, 0}};
        for (int j = 0; j < np; ++j) {
            uchar4 pj = data[w * pxls[j].y + pxls[j].x];
            float3 diff = make_float3(0, 0, 0);
            diff.x = pj.x - mean.x;
            diff.y = pj.y - mean.y;
            diff.z = pj.z - mean.z;
            cov[0][0] += diff.x * diff.x;
            cov[0][1] += diff.x * diff.y;
            cov[0][2] += diff.x * diff.z;
            cov[1][0] += diff.y * diff.x;
            cov[1][1] += diff.y * diff.y;
            cov[1][2] += diff.y * diff.z;
            cov[2][0] += diff.z * diff.x;
            cov[2][1] += diff.z * diff.y;
            cov[2][2] += diff.z * diff.z;
        }
        if (np > 1) {
            cov[0][0] *= 1.0 / (np - 1);
            cov[0][1] *= 1.0 / (np - 1);
            cov[0][2] *= 1.0 / (np - 1);
            cov[1][0] *= 1.0 / (np - 1);
            cov[1][1] *= 1.0 / (np - 1);
            cov[1][2] *= 1.0 / (np - 1);
            cov[2][0] *= 1.0 / (np - 1);
            cov[2][1] *= 1.0 / (np - 1);
            cov[2][2] *= 1.0 / (np - 1);
        }
        inverse33(cov);
        copy_matrix33(cov_arr[i], cov);
    }
    free(pxls);
    CSC(cudaMemcpyToSymbol(avgGPU, mean_arr, sizeof(TVector) * MAX_CLASSES, 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(covGPU, cov_arr, sizeof(TMatrix) * MAX_CLASSES, 0, cudaMemcpyHostToDevice));

    uchar4 *dev_arr;
    CSC(cudaMalloc(&dev_arr, w * sizeof(uchar4) * h));
    CSC(cudaMemcpy(dev_arr, data, w * sizeof(uchar4) * h, cudaMemcpyHostToDevice));


    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    kernel << < 128, 128 >> > (dev_arr, w, h, nc);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    fprintf(stderr, "time = %f\n", time);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);



    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_arr, w * sizeof(uchar4) * h, cudaMemcpyDeviceToHost));

    FILE *file_out = fopen(output_file, "wb");
    fwrite(&w_int, sizeof(int), 1, file_out);
    fwrite(&h_int, sizeof(int), 1, file_out);
//    printf("%lu %lu -- %lu", w, h, w * h * sizeof(uchar4));
    fwrite(data, sizeof(uchar4), w * h, file_out);
    fclose(file_out);
    CSC(cudaFree(dev_arr));
    free(data);
    return 0;
}
