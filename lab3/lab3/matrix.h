#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdlib.h>
#include "image.h"


typedef struct Matrix {
    double data[3][3];

    __host__ __device__ void setAllToZero() {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                data[i][j] = 0.0;
    }

    __host__ __device__ void makeIdentity() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == j) {
                    data[i][j] = 1.0;
                } else {
                    data[i][j] = 0.0;
                }
            }
        }
    }

    __host__ __device__ Matrix inverse() {
        Matrix tmp;
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                tmp.data[i][j] = data[i][j];
            }
        }
        Matrix e;
        e.makeIdentity();
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                if (tmp.data[j][j] == 0.0)
                    continue;
                if (i == j) {
                    double ratio = tmp.data[j][j];

                    for (int k = 0; k < 3; ++k) {
                        tmp.data[i][k] /= ratio;
                        e.data[i][k] /= ratio;
                    }
                } else {
                    double ratio = tmp.data[i][j] / tmp.data[j][j];

                    for (int k = 0; k < 3; ++k) {
                        tmp.data[i][k] -= tmp.data[j][k] * ratio;
                        e.data[i][k] -= e.data[j][k] * ratio;
                    }
                }
            }
        }
        return e;
    }

    __host__ __device__ void multipleByNumber(double lambda) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                data[i][j] *= lambda;
    }

    __host__ __device__ void addMatrix(Matrix m) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                data[i][j] += m.data[i][j];
    }

    __host__ __device__ void subtractMatrix(Matrix m) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                data[i][j] -= m.data[i][j];
    }

    __host__ __device__ void print() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                printf("%f\t", data[i][j]);
            printf("\n"); 
        }
    }

} Matrix;

#endif