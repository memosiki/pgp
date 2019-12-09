#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "image.h"

typedef struct Vector {
    double data[3];

    __host__ __device__ void setAllToZero() {
        for (int i = 0; i < 3; i++)
            data[i] = 0.0;
    }

    __host__ __device__ void createFromPixel(Pixel p) {
        data[0] = p.x;
        data[1] = p.y;
        data[2] = p.z;
    }

    __host__ __device__ double multipleByVector(Vector v) {
        double res = 0.0;
        for (int i = 0; i < 3; i++)
            res += data[i] * v.data[i];
        return res;
    }

    __host__ __device__ Matrix multipleByVectorTransposed(Vector v) {
        Matrix m;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m.data[i][j] = data[i] * v.data[j];
        return m;
    }

    __host__ __device__ void addVector(Vector v) {
        for (int i = 0; i < 3; i++)
            data[i] += v.data[i];
    }

    __host__ __device__ void subtractVector(Vector v) {
        for (int i = 0; i < 3; i++)
            data[i] -= v.data[i];
    }

    __host__ __device__ void multipleByNumber(double lambda) {
        for (int i = 0; i < 3; i++)
            data[i] *= lambda;
    }

    __host__ __device__ Vector multipleByMatrix(Matrix m) {
        Vector res;
        res.setAllToZero();
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                res.data[j] += data[k] * m.data[k][j];
        return res;
    }

    __host__ __device__ void print() {
        for (int i = 0; i < 3; i++)
            printf("%f\t", data[i]);
        printf("\n"); 
    }

} Vector;

#endif