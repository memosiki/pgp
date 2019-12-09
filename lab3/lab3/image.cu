#include "image.h"

Image * createImage(int width, int height) {
    Image *image = mallocImage();
    image->width = width;
    image->height = height;
    image->pixels = mallocPixels(getPixelsCount(image));
    return image;
}

Image * mallocImage() {
    return (Image*)malloc(sizeof(Image));
}


Pixel * mallocPixels(int length) {
    return (Pixel*)malloc(sizeof(Pixel) * length);
}

Image * readImageFromFile(const char *filename) {
    Image *resultImage = mallocImage();
    FILE *in = fopen(filename, "rb");
    setImageDimensionsFromFile(resultImage, in);
    setImagePixelsFromFile(resultImage, in);
    fclose(in);
    return resultImage;
}

void setImageDimensionsFromFile(Image *image, FILE *in) {
    fread(&image->width, sizeof(image->width), 1, in);
    fread(&image->height, sizeof(image->height), 1, in);
}

void setImagePixelsFromFile(Image *image, FILE *in) {
    int count = getPixelsCount(image);
    image->pixels = mallocPixels(count);
    fread(image->pixels, sizeof(Pixel), count, in);
}

void writeImageToFile(Image *image, const char *filename) {
    FILE *out = fopen(filename, "wb");
    fwrite(&image->width, sizeof(image->width), 1, out);
    fwrite(&image->height, sizeof(image->height), 1, out);
    fwrite(image->pixels, sizeof(Pixel), getPixelsCount(image), out);
    fclose(out);
}

int getPixelsCount(Image * image) {
    return image->width * image->height;
}

Pixel getPixel(Image *image, int x, int y) {
    int idx = image->width * y + x;
    return image->pixels[idx];
}

void deleteImage(Image *image) {
    free(image->pixels);
    image = NULL;
}