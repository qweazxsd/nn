#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"
int main (int argc, char *argv[])
{
    Mat a = matrix_alloc(5, 3);
    float arr[] = {
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3,
        4, 4, 4,
    };
    a.ele = arr;
    srand(time(NULL));
    matrix_shuffle_rows(a);
    MAT_PRINT(a);
    matrix_shuffle_rows(a);
    MAT_PRINT(a);
    return 0;
}
        

