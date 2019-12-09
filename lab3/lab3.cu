#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <algorithm>

#define NAMELEN 100
#define NUMCLASSES 2


#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)


__constant__ float3 meanGPU[2];
__constant__ float covGpu[NUMCLASSES * 9];

__device__ int get_dist (uchar4 p, int j){
    float cov[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; //inverse cov
    float3 mean = meanGPU[j];
    float3 diff = make_float3(0, 0, 0);
    float3 c; // for matmul

    for (int k = 0; k < 9; ++k) {
        cov[k] =  covGpu[j * 9 + k];
    }
    diff.x = float(p.x) - mean.x;
    diff.y = p.y - mean.y;
    diff.z = p.z - mean.z;
//    float a = cov[0];
//    printf("mean %d %d %d \n", mean.x, mean.y, mean.z);

    c.x = diff.x * cov[0 * 3 + 0] + diff.y * cov[1 * 3 + 0] + diff.z * cov[2 * 3 + 0];
    c.y = diff.x * cov[0 * 3 + 1] + diff.y * cov[1 * 3 + 1] + diff.z * cov[2 * 3 + 1];
    c.z = diff.x * cov[0 * 3 + 2] + diff.y * cov[1 * 3 + 2] + diff.z * cov[2 * 3 + 2];

    float dist = c.x * diff.x + c.y * diff.y + c.z * diff.z;
    return dist;
}

__global__ void kernel(uchar4 *dev_arr, int w, int h, int nc) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    float dist, new_dist; // Mahalanobis dist
    int cls = 0; // class
    uchar4 p;

    for (int i = idx; i < w * h; i += offset) {
        p = dev_arr[i];
        dist = -1e10;
        for (int j = 0; j < nc; ++j) {
            new_dist = get_dist(p, j);
            if (new_dist > dist) {
                dist = new_dist;
                cls = j;
            }

        }
        dev_arr[i].w = cls;
    }
}



void read_data(char *inname, uchar4 **data, int *w, int *h) {

}


void write_data(char *outname, uchar4 *data, int w, int h) {
    FILE *fout = fopen(outname, "wb");
    fwrite(&w, sizeof(int), 1, fout);
    fwrite(&h, sizeof(int), 1, fout);
    fwrite(data, sizeof(uchar4), w * h, fout);
    fclose(fout);
}


int main() {
    char inname[NAMELEN];
    char outname[NAMELEN];
    int w, h;
    int nc, np, x, y;
    uchar4 *data;
    float3 mean_arr[NUMCLASSES] = {0};

    scanf("%s", inname);
    scanf("%s", outname);
    read_data(inname, &data, &w, &h);

    uchar2 *pxls = (uchar2 *) malloc(sizeof(uchar4) * 2 * 524288);
    float cov_arr[NUMCLASSES * 9] = {0};
    scanf("%d", &nc);
    for (int i = 0; i < nc; ++i) {
        scanf("%d", &np);

        float3 mean = make_float3(0, 0, 0);
        for (int j = 0; j < np; ++j) {
            scanf("%d%d", &x, &y);
            pxls[j].x = x;
            pxls[j].y = y;
            uchar4 pj = data[y * w + x];
            mean.x += pj.x;
            mean.y += pj.y;
            mean.z += pj.z;
        }
//        std::cout << "mean :" << mean.x << " " << mean.y << " " << mean.z << std::endl;

        mean.x /= np;
        mean.y /= np;
        mean.z /= np;
        mean_arr[i] = mean;
        float cov[9] = {0};
        for (int j = 0; j < np; ++j) {
            uchar4 pj = data[pxls[j].y * w + pxls[j].x];
            float3 diff = make_float3(0, 0, 0);
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

        float det = cov[0 * 3 + 0] * (cov[1 * 3 + 1] * cov[2 * 3 + 2] - cov[2 * 3 + 1] * cov[1 * 3 + 2]) -
                    cov[0 * 3 + 1] * (cov[1 * 3 + 0] * cov[2 * 3 + 2] - cov[2 * 3 + 0] * cov[1 * 3 + 2]) +
                    cov[0 * 3 + 2] * (cov[1 * 3 + 0] * cov[2 * 3 + 1] - cov[2 * 3 + 0] * cov[1 * 3 + 1]);
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

//        std::cout << "det " << det << std::endl;
//        std::cout << cov_arr[i * 9 + 0] << " " <<
//                  cov_arr[i * 9 + 1] << " " <<
//                  cov_arr[i * 9 + 2] << " " <<
//                  cov_arr[i * 9 + 3] << " " <<
//                  cov_arr[i * 9 + 4] << " " <<
//                  cov_arr[i * 9 + 5] << " " <<
//                  cov_arr[i * 9 + 6] << " " <<
//                  cov_arr[i * 9 + 7] << " " <<
//                  cov_arr[i * 9 + 8] << " " << std::endl;
//
    }

    CSC(cudaMemcpyToSymbol(meanGPU, mean_arr, sizeof(float3) * 2, 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(covGpu, cov_arr, sizeof(float) * NUMCLASSES * 9, 0, cudaMemcpyHostToDevice));
    CSC(cudaGetLastError());

    uchar4 *dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(uchar4) * h * w));
    CSC(cudaMemcpy(dev_arr, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));

    kernel <<< 16, 16 >>> (dev_arr, w, h, nc);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_arr, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost));

    write_data(outname, data, w, h);

    CSC(cudaFree(dev_arr));
    free(data);

    return 0;
}