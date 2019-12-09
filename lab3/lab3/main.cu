#include <stdio.h>
#include <string.h>
#include "image.h"
#include "classifier.h"
#include "kernel.h"
#include "csc.h"

int classify() {
    char inputFile[256];
    char outputFile[256];
    scanf("%s", inputFile);
    scanf("%s", outputFile);
    Image *inputImage = readImageFromFile(inputFile);
    Pixel* outputPixels;
    int size = sizeof(Pixel) * inputImage->width * inputImage->height;
    CSC(cudaMalloc(&outputPixels, size));
    CSC(cudaMemcpy(outputPixels, inputImage->pixels, size, cudaMemcpyHostToDevice));
    Classifier *c = createClassifier(inputImage);
    copyClassifierToConstant(c);
    dim3 gridSize(16, 16);
	dim3 blockSize(16, 16);
	classifyPixels<<<gridSize, blockSize>>>(outputPixels, inputImage->width, inputImage->height);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(inputImage->pixels, outputPixels, size, cudaMemcpyDeviceToHost));
    writeImageToFile(inputImage, outputFile);
    deleteClassifier(c);
    deleteImage(inputImage);
    return 0;
}

int main(void)
{
    classify();
	return 0;
}