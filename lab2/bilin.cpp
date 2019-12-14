#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)


//texture<uchar4, 2, cudaReadModeElementType> tex;

//
//__global__ void kernel(uchar4 *dev_arr, int w, int h, int w_new, int h_new) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    int offset = blockDim.x * gridDim.x;
//
//    int i, j;  // current pixel coords
//    float x, y; // float coords of pixel in original image
//    int x0, y0, x1, y1; // coords of points of interpolation
//    float wa, wb, wc, wd; // weights of points of interpolation
//    uchar4 Ia, Ib, Ic, Id; // points of interpolation
//    uchar4 p; // result
//
//    while (idx < w_new * h_new) {
//
//        i = idx % w_new;
//        j = idx / w_new;
//        x = (i + 0.5) * (float(w) / w_new) - 0.5;
//        y = (j + 0.5) * (float(h) / h_new) - 0.5;
//
//        x0 = floorf(x);
//        y0 = floorf(y);
//        x1 = x0 + 1;
//        y1 = y0 + 1;
//
//        Ia = tex2D(tex, x0, y0);
//        Ib = tex2D(tex, x0, y1);
//        Ic = tex2D(tex, x1, y0);
//        Id = tex2D(tex, x1, y1);
//
//
//        wa = (x1 - x) * (y1 - y);
//        wb = (x1 - x) * (y - y0);
//        wc = (x - x0) * (y1 - y);
//        wd = (x - x0) * (y - y0);
//
//        p.x = floorf(wa * Ia.x + wb * Ib.x + wc * Ic.x + wd * Id.x);
//        p.y = floorf(wa * Ia.y + wb * Ib.y + wc * Ic.y + wd * Id.y);
//        p.z = floorf(wa * Ia.z + wb * Ib.z + wc * Ic.z + wd * Id.z);
//        p.w = 0;
//
//        dev_arr[idx] = p;
//        idx += offset;
//    }
//}


void read_image(char *filename, uchar4 **data, int *w, int *h) {
    FILE *file_in = fopen(filename, "rb");
    fread(w, sizeof(int), 1, file_in);
    fread(h, sizeof(int), 1, file_in);
    *data = (uchar4 *) malloc(sizeof(uchar4) * (*h) * (*w));
    fread(*data, sizeof(uchar4), (*h) * (*w), file_in);
    fclose(file_in);
}


void write_image(char *filename, uchar4 *data, int w, int h) {
    FILE *file_out = fopen(filename, "wb");
    fwrite(&w, sizeof(int), 1, file_out);
    fwrite(&h, sizeof(int), 1, file_out);
    fwrite(data, sizeof(uchar4), w * h, file_out);
    fclose(file_out);
}

//
//cudaArray *initialize_texture(uchar4 *data, int w, int h) {
//    cudaArray *texture_arr;
//    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
//    CSC(cudaMallocArray(&texture_arr, &ch, w, h));
//    CSC(cudaMemcpyToArray(texture_arr, 0, 0, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));
//
//    tex.addressMode[0] = cudaAddressModeClamp;
//    tex.addressMode[1] = cudaAddressModeClamp;
//    tex.channelDesc = ch;
//    tex.filterMode = cudaFilterModePoint;
//    tex.normalized = false;
//
//    CSC(cudaBindTextureToArray(tex, texture_arr, ch));
//
//    return texture_arr;
//}

uchar4 getp(uchar4 arr[], int x, int y, int w, int h) {
//    printf("%d %d from %d  %d", x, y, y*w+x, w*h);

    if (x < 0)
        x = 0;
    if (x >= w)
        x = w - 1;
    if (y < 0)
        y = 0;
    if (y >= h)
        y = h - 1;
    if (y * w + x >= w * h)
        printf("ggggggg");
    return arr[y * w + x];
}

int main() {
    char in_name[100];
    char out_name[100];
    int w, h;
    int w_new, h_new;
    uchar4 *data;

    scanf("%s", in_name);
    scanf("%s", out_name);
    scanf("%d", &w_new);
    scanf("%d", &h_new);

    read_image(in_name, &data, &w, &h);


    uchar4 *new_data = (uchar4 *) realloc(data, sizeof(uchar4) * w_new * h_new);

//    kernel << < 1, 32 >> > (dev_arr, w, h, w_new, h_new);

    clock_t begin = clock();
    static struct timeval tm1;
    ///////////////////////////
    int idx = 0;

    int i, j;  // current pixel coords
    float x, y; // float coords of pixel in original image
    int x0, y0, x1, y1; // coords of points of interpolation
    float wa, wb, wc, wd; // weights of points of interpolation
    uchar4 Ia, Ib, Ic, Id; // points of interpolation
    uchar4 p; // result

    while (idx < w_new * h_new) {
        printf("%d", idx);

        i = idx % w_new;
        j = idx / w_new;
        x = (i + 0.5) * (float(w) / w_new) - 0.5;
        y = (j + 0.5) * (float(h) / h_new) - 0.5;

        x0 = floorf(x);
        y0 = floorf(y);
        x1 = x0 + 1;
        y1 = y0 + 1;

        Ia = getp(data, x0, y0, w, h);
        Ib = getp(data, x0, y1, w, h);
        Ic = getp(data, x1, y0, w, h);
        Id = getp(data, x1, y1, w, h);


        wa = (x1 - x) * (y1 - y);
        wb = (x1 - x) * (y - y0);
        wc = (x - x0) * (y1 - y);
        wd = (x - x0) * (y - y0);

        p.x = floorf(wa * Ia.x + wb * Ib.x + wc * Ic.x + wd * Id.x);
        p.y = floorf(wa * Ia.y + wb * Ib.y + wc * Ic.y + wd * Id.y);
        p.z = floorf(wa * Ia.z + wb * Ib.z + wc * Ic.z + wd * Id.z);
        p.w = 0;

        new_data[idx] = p;
        idx += 1;
    }
    /////////////////////
//    int in = 0;
//    while (in != - 1)
//        in++;
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(stderr, "time = %lf\n", elapsed_secs);

    struct timeval tm2;
    gettimeofday(&tm2, NULL);
//    unsigned long long t = 1000 * (tm2.tv_sec - tm1.tv_sec) + (tm2.tv_usec - tm1.tv_usec) / 1000;
//    printf("%llu ms\n", t);


    write_image(out_name, data, w_new, h_new);

//    printf("a\n");
//    free(data);
//    free(new_data);
    return 0;
}