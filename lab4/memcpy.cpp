#include <numeric>
#include <stdio.h>
#include <string.h>

int a[4][4];
// int b[10];
int main(int argc, char const *argv[]) {
    std::iota((int*)a, (int*)a+16, 0);
    // std::iota(b, b+10, 0);

    memcpy((int*)a, (int*)a+5, 5*sizeof(int));
    for (size_t i = 0; i < 16; i++) {
        printf("%d ", ((int*)a)[i]);
    }
    printf("\n");
    return 0;
}
