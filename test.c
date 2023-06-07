#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb_image_write.h"

int main (int argc, char *argv[])
{
    FILE *fp;
    //FILE *wfp;
    int c;

    fp = fopen("mnist/test_img", "rb");
    if (fp==NULL) {
        printf("Could not open file");
        return 1;
    }

    //wfp = fopen("data.c", "w");
    //if (wfp==NULL) {
    //    printf("Could not open file");
    //    return 1;
    //}

    uint8_t magic[4];
    for (size_t i = 0; i < 4 && (c=getc(fp)) != EOF; ++i) {
        magic[i] = c;
    }
    
    uint32_t npics = 0;
    for (size_t i = 0; i < 4 && (c=getc(fp)) != EOF; ++i) {
        npics |= ((uint32_t) c) << (8*(4-i-1));
    }

    uint32_t rows = 0;
    for (size_t i = 0; i < 4 && (c=getc(fp)) != EOF; ++i) {
        rows |= ((uint32_t) c) << (8*(4-i-1));
    }

    uint32_t cols = 0;
    for (size_t i = 0; i < 4 && (c=getc(fp)) != EOF; ++i) {
        cols |= ((uint32_t) c) << (8*(4-i-1));
    }

    //fprintf(wfp, "unsigned char a[] = {\n");
    uint8_t pixels[rows*cols];
    char buf[256];
    size_t x;
    for (size_t j = 0; j < rows*cols*2 && (c=getc(fp)) != EOF; ++j) {
        pixels[j%(rows*cols)] = c;
        printf("%d ", c);
        x = (j+1)%(rows*cols); 
        //fprintf(wfp, "%d, ", c);
        if (x==0) {
            //fprintf(wfp, "\n");
            //snprintf(buf, sizeof(buf), "test_img/output%lu.png", (j+1)/(rows*cols));
            //stbi_write_png(buf, rows, cols, 1, pixels, rows*sizeof(*pixels));
        }
        if ((j%rows==0)&&(j>0)) {
            printf("\n");
        }
    }
    

    //fprintf(wfp, "};");
    //fclose(wfp);
    fclose(fp);
    
    return 0;
}
        

