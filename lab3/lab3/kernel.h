#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "image.h"
#include "matrix.h"
#include "vector.h"
#include "classifier.h"
#include "csc.h"

__device__ double calcMahalanobisDistance(Pixel* pixel, int i);
__global__ void classifyPixels(Pixel* pixels, int w, int h);

#endif