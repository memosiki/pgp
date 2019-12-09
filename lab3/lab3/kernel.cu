#include "kernel.h"

__device__ double calcMahalanobisDistance(Pixel pixel, int i) {
    Vector v;
    Vector tmp;
    v.createFromPixel(pixel);
    v.subtractVector(avgs[i]);
    tmp = v.multipleByMatrix(covs[i]);
    double res = v.multipleByVector(tmp);
    return res * (-1.0);
}

__global__ void classifyPixels(Pixel* pixels, int w, int h) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetY = gridDim.y * blockDim.y;
	int offsetX = gridDim.x * blockDim.x;

	for (int x = idx; x < w; x += offsetX)
	{
		for (int y = idy; y < h; y += offsetY)
		{
            int offset = y * w + x;
			double maxValue = calcMahalanobisDistance(pixels[offset], 0);
            int indexOfMaxValue = 0;

			for (int k = 1; k < nc; ++k)
			{
				double tmp = calcMahalanobisDistance(pixels[offset], k);
				if (tmp > maxValue)
				{
					maxValue = tmp;
					indexOfMaxValue = k;
                }
			}

            pixels[offset].w = (unsigned char)indexOfMaxValue;
        }
	}
}