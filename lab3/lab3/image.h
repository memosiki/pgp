#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <stdlib.h>

typedef uchar4 Pixel;
typedef struct {
    int width;
    int height;
    Pixel *pixels;
} Image;


Image * mallocImage();
Pixel * mallocPixels(int length);
int getPixelsCount(Image * image);
Image * createImage(int width, int height);
void setImageDimensionsFromFile(Image *image, FILE *in);
void setImagePixelsFromFile(Image *image, FILE *out);
Image * readImageFromFile(const char *filename);
void writeImageToFile(Image *image, const char *filename);
void deleteImage(Image *image);
Pixel getPixel(Image *image, int x, int y);
#endif