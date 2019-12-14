#include <stdio.h>
#include <algorithm>
#include <float.h>

#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)

typedef double TMatrix[3][3];
typedef double3 TVector;
const int MAX_CLASSES = 32;

__constant__ TVector
avgGPU[MAX_CLASSES];
__constant__ TMatrix covGPU[MAX_CLASSES];

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
__host__ double determ33(TMatrix cov) {
    double det  = cov[0 ][ 0] * (cov[1 ][ 1] * cov[2 ][ 2] - cov[2 ][ 1] * cov[1 ][ 2]) -
                     cov[0 ][ 1] * (cov[1 ][ 0] * cov[2 ][ 2] - cov[2 ][ 0] * cov[1 ][ 2]) +
                     cov[0 ][ 2] * (cov[1 ][ 0] * cov[2 ][ 1] - cov[2 ][ 0] * cov[1 ][ 1]);
    return det;
}
__host__ void inverse33(TMatrix a) {
    double det = determ33(a);
    a[0][0] = 1 / det * (a[1][1] * a[2][2] - a[2][1] * a[1][2]);
    a[0][1] = -1 / det * (a[1][0] * a[2][2] - a[2][0] * a[1][2]);
    a[0][2] = 1 / det * (a[1][0] * a[2][1] - a[2][0] * a[1][1]);
    a[1][0] = -1 / det * (a[0][1] * a[2][2] - a[2][1] * a[0][2]);
    a[1][1] = 1 / det * (a[0][0] * a[2][2] - a[2][0] * a[0][2]);
    a[1][2] = -1 / det * (a[0][0] * a[2][1] - a[2][0] * a[0][1]);
    a[2][0] = 1 / det * (a[0][1] * a[1][2] - a[1][1] * a[0][2]);
    a[2][1] = -1 / det * (a[0][0] * a[1][2] - a[1][0] * a[0][2]);
    a[2][2] = 1 / det * (a[0][0] * a[1][1] - a[1][0] * a[0][1]);
    std::swap(a[0][1], a[1][0]);
    std::swap(a[0][2], a[2][0]);
    std::swap(a[1][2], a[2][1]);
}

__device__ double get_dist(uchar4 p, int j) {
    double cov[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; //inverse cov
    double3 mean = avgGPU[j];
    double3 diff = make_double3(0, 0, 0);
    double3 c; // for matmul

    for (int k = 0; k < 9; ++k)
        cov[k] = covGPU[j * 9 + k];

    diff.x = p.x - mean.x;
    diff.y = p.y - mean.y;
    diff.z = p.z - mean.z;

    c.x = diff.x * cov[0 * 3 + 0] + diff.y * cov[1 * 3 + 0] + diff.z * cov[2 * 3 + 0];
    c.y = diff.x * cov[0 * 3 + 1] + diff.y * cov[1 * 3 + 1] + diff.z * cov[2 * 3 + 1];
    c.z = diff.x * cov[0 * 3 + 2] + diff.y * cov[1 * 3 + 2] + diff.z * cov[2 * 3 + 2];

    double dist = c.x * diff.x + c.y * diff.y + c.z * diff.z;
    dist = -dist;
    return dist;
}

__global__ void kernel(uchar4 *dev_arr, int w, int h, int nc) {
    long long idx = (long long) threadIdx.x + blockIdx.x * blockDim.x;
    long long offset = (long long) blockDim.x * gridDim.x;
    double dist, new_dist; // Mahalanobis dist
    int cls = nc - 1; // class
    uchar4 p;

    while (idx <  (long long) w * h) {
//    for (long long i = idx; i < (long long) w * h; i += offset) {
        p = dev_arr[idx];
        dist = -DBL_MAX;
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
    int w, h;
    int nc, np, x, y;
    uchar4 *data;
    double3 mean_arr[MAX_CLASSES];
    double cov_arr[MAX_CLASSES * 9] = {0};
    uchar2 *pxls = (uchar2 *) malloc(sizeof(uchar4) * 2 * 524288); // 2**19=524288

    scanf("%s", input_file);
    FILE *file_in = fopen(input_file, "rb");
    fread(&w, sizeof(int), 1, file_in);
    fread(&h, sizeof(int), 1, file_in);
    data = (uchar4 *) malloc((long long) sizeof(uchar4) * h * w);
    fread(data, sizeof(uchar4), (long long) h * w, file_in);
    fclose(file_in);

    scanf("%s", output_file);
    scanf("%d", &nc);
    for (int i = 0; i < nc; ++i) {
        scanf("%d", &np);

        double3 mean = make_double3(0, 0, 0);
        for (int j = 0; j < np; ++j) {
            scanf("%d%d", &x, &y);
            pxls[j].x = x;
            pxls[j].y = y;
            uchar4 pj = data[(long long)y * w + x];
            mean.x += pj.x;
            mean.y += pj.y;
            mean.z += pj.z;
        }

        mean.x /= np;
        mean.y /= np;
        mean.z /= np;
        mean_arr[i] = mean;
        double cov[9] = {0};
        for (int j = 0; j < np; ++j) {
            uchar4 pj = data[(long long) pxls[j].y * w + pxls[j].x];
            double3 diff = make_double3(0, 0, 0);
            diff.x = pj.x - mean.x;
            diff.y = pj.y - mean.y;
            diff.z = pj.z - mean.z;
            cov[0] += diff.x * diff.x;
            cov[1] += diff.x * diff.y;
            cov[2] += diff.x * diff.z;
            cov[3] += diff.y * diff.x;
            cov[4] += diff.y * diff.y;
            cov[5] += diff.y * diff.z;
            cov[6] += diff.z * diff.x;
            cov[7] += diff.z * diff.y;
            cov[8] += diff.z * diff.z;

        }
        cov[0] /= np - 1;
        cov[1] /= np - 1;
        cov[2] /= np - 1;
        cov[3] /= np - 1;
        cov[4] /= np - 1;
        cov[5] /= np - 1;
        cov[6] /= np - 1;
        cov[7] /= np - 1;
        cov[8] /= np - 1;

        // inverse
        cov_arr[i * 9 + 0 * 3 + 0] = 1 / det * (cov[1 * 3 + 1] * cov[2 * 3 + 2] - cov[2 * 3 + 1] * cov[1 * 3 + 2]);
        cov_arr[i * 9 + 0 * 3 + 1] = -1 / det * (cov[1 * 3 + 0] * cov[2 * 3 + 2] - cov[2 * 3 + 0] * cov[1 * 3 + 2]);
        cov_arr[i * 9 + 0 * 3 + 2] = 1 / det * (cov[1 * 3 + 0] * cov[2 * 3 + 1] - cov[2 * 3 + 0] * cov[1 * 3 + 1]);
        cov_arr[i * 9 + 1 * 3 + 0] = -1 / det * (cov[0 * 3 + 1] * cov[2 * 3 + 2] - cov[2 * 3 + 1] * cov[0 * 3 + 2]);
        cov_arr[i * 9 + 1 * 3 + 1] = 1 / det * (cov[0 * 3 + 0] * cov[2 * 3 + 2] - cov[2 * 3 + 0] * cov[0 * 3 + 2]);
        cov_arr[i * 9 + 1 * 3 + 2] = -1 / det * (cov[0 * 3 + 0] * cov[2 * 3 + 1] - cov[2 * 3 + 0] * cov[0 * 3 + 1]);
        cov_arr[i * 9 + 2 * 3 + 0] = 1 / det * (cov[0 * 3 + 1] * cov[1 * 3 + 2] - cov[1 * 3 + 1] * cov[0 * 3 + 2]);
        cov_arr[i * 9 + 2 * 3 + 1] = -1 / det * (cov[0 * 3 + 0] * cov[1 * 3 + 2] - cov[1 * 3 + 0] * cov[0 * 3 + 2]);
        cov_arr[i * 9 + 2 * 3 + 2] = 1 / det * (cov[0 * 3 + 0] * cov[1 * 3 + 1] - cov[1 * 3 + 0] * cov[0 * 3 + 1]);
        std::swap(cov_arr[i * 9 + 0 * 3 + 1], cov_arr[i * 9 + 1 * 3 + 0]);
        std::swap(cov_arr[i * 9 + 0 * 3 + 2], cov_arr[i * 9 + 2 * 3 + 0]);
        std::swap(cov_arr[i * 9 + 1 * 3 + 2], cov_arr[i * 9 + 2 * 3 + 1]);
    }
    free(pxls);
    CSC(cudaMemcpyToSymbol(avgGPU, mean_arr, sizeof(double3) * MAX_CLASSES, 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(covGPU, cov_arr, sizeof(double) * MAX_CLASSES * 9, 0, cudaMemcpyHostToDevice));

    uchar4 *dev_arr;
    CSC(cudaMalloc(&dev_arr, (long long) sizeof(uchar4) * h * w));
    CSC(cudaMemcpy(dev_arr, data, (long long) sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));

    kernel << < 256, 256 >> > (dev_arr, w, h, nc);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_arr, (long long) sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost));

    FILE *file_out = fopen(output_file, "wb");
    fwrite(&w, sizeof(int), 1, file_out);
    fwrite(&h, sizeof(int), 1, file_out);
    fwrite(data, (long long) sizeof(uchar4), w * h, file_out);
    fclose(file_out);
    CSC(cudaFree(dev_arr));
    free(data);
    return 0;
}