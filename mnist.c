#include <math.h>
#include <raylib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define NN_IMPLEMENTATION
#include "nn.h"

void nn_process_batch(NN nn, Mat t, float rate, size_t current_batch, size_t batch_size, float* cost) {
    Mat batch_in;
    Mat batch_out;
    if (t.rows-current_batch*batch_size < batch_size) {
        batch_in.rows  = t.rows-current_batch*batch_size;
        batch_out.rows = t.rows-current_batch*batch_size;
    }
    else {
        batch_in.rows  = batch_size;
        batch_out.rows = batch_size;
    }
    batch_in.cols    = NN_INPUT(nn).rows;
    batch_out.cols   = NN_OUTPUT(nn).rows;
    batch_in.stride  = t.stride;
    batch_out.stride = t.stride;

    //batches are just a view to the training input&output 
    //and these are just pointers to the first element of the matrix
    batch_in.ele  = &MAT_ELE(t, current_batch*batch_size, 0);
    batch_out.ele = &MAT_ELE(t, current_batch*batch_size, batch_in.cols);
    nn_backprop(nn, batch_in, batch_out, rate, t.rows);
    *cost += nn_sum_cost(nn, batch_in, batch_out)/t.rows;

}
//---------------------------------------------------
//---------------- HYPER PARAMETERS -----------------
//---------------------------------------------------
FILE *test_img_fp, *test_lbl_fp, *train_img_fp, *train_lbl_fp;
char test_img_fn[] = "data/test_img";
char test_lbl_fn[] = "data/test_lbl";
char train_img_fn[] = "data/train_img";
char train_lbl_fn[] = "data/train_lbl";


#define FPS 60
#define WIN_WIDTH_RATIO 16
#define WIN_LENGTH_RATIO 9
#define WIN_SCALE 100
#define MAX_EPOCHS 100
#define EPOCHS_PER_FRAME 1 
#define RATE 1e-2 
size_t batch_s = 10;
int paused = false;
//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

int main (int argc, char *argv[])
{
    //---------------------------------------------------------
    //            LOADING TRAINING AND TESTING DATA
    //---------------------------------------------------------
    int trnib,tstib,trnlblb,tstlblb;  //variables to hold the current byte of each file

    //OPENING FILES
    train_img_fp = fopen(train_img_fn, "rb");
    if (train_img_fp==NULL) {
        printf("ERROR: Could not open file %s", train_img_fn);
        exit(1);
    }
    test_img_fp = fopen(test_img_fn, "rb");
    if (test_img_fp==NULL) {
        printf("ERROR: Could not open file %s", test_img_fn);
        exit(1);
    }
    train_lbl_fp = fopen(train_lbl_fn, "rb");
    if (train_lbl_fp==NULL) {
        printf("ERROR: Could not open file %s", train_lbl_fn);
        exit(1);
    }
    test_lbl_fp = fopen(test_lbl_fn, "rb");
    if (test_lbl_fp==NULL) {
        printf("ERROR: Could not open file %s", test_lbl_fn);
        exit(1);
    }

    //READING THE MAGIC OF ALL FILES
    uint8_t magic[4];
    for (size_t i = 0; i < 4 && (trnib=getc(train_img_fp)) != EOF && (tstib=getc(test_img_fp)) != EOF && (trnlblb=getc(train_lbl_fp)) != EOF && (tstlblb=getc(test_lbl_fp)) != EOF; ++i) {
        magic[i] = trnib;
    }

    //GET NUMBER OF IMAGES FROM FILES
    uint32_t n_train_img = 0;
    uint32_t n_test_img  = 0;
    uint32_t n_train_lbl = 0;
    uint32_t n_test_lbl  = 0;
    for (size_t i = 0; i < 4 && (trnib=getc(train_img_fp)) != EOF && (tstib=getc(test_img_fp)) != EOF && (trnlblb=getc(train_lbl_fp)) != EOF && (tstlblb=getc(test_lbl_fp)) != EOF; ++i) {
        n_train_img |= ((uint32_t) trnib)   << (8*(4-i-1));
        n_test_img  |= ((uint32_t) tstib)   << (8*(4-i-1));
        n_train_lbl |= ((uint32_t) trnlblb) << (8*(4-i-1));
        n_test_lbl  |= ((uint32_t) tstlblb) << (8*(4-i-1));
    }

    //GET IMAGE SIZE FROM FILES
    uint32_t img_w = 0;
    for (size_t i = 0; i < 4 && (trnib=getc(train_img_fp)) != EOF && (tstib=getc(test_img_fp)) != EOF; ++i) {
        img_w |= ((uint32_t) trnib) << (8*(4-i-1));
    }

    uint32_t img_h = 0;
    for (size_t i = 0; i < 4 && (trnib=getc(train_img_fp)) != EOF && (tstib=getc(test_img_fp)) != EOF; ++i) {
        img_h |= ((uint32_t) trnib) << (8*(4-i-1));
    }

    /*
    To use stochastic gradient decent the order of the inputs need to be randimized.
    To keep the inputs aligned with the outputs during the randomization process 
    the inputs and the outputs need to be as one matrix.
    */

    Mat train = matrix_alloc(n_train_img, img_w*img_h+10);
    Mat test  = matrix_alloc(n_test_img, img_w*img_h+10); 

    // LOADING TRAINING IMAGES TO MATRIX
    for (size_t i = 0; i < n_train_img; ++i) {
        for (size_t j = 0; j < img_w*img_h; ++j) {
            if ((trnib=getc(train_img_fp)) == EOF) {
                printf("ERROR: Reached EOF on file %s", train_img_fn);
                exit(1);
            }
            MAT_ELE(train, i, j) = (float) trnib/255.f;
        }
    }
    // LOADING TRAINING LABELS TO MATRIX
    for (size_t i = 0; i < n_train_lbl; ++i) {
        if ((trnlblb=getc(train_lbl_fp)) == EOF) {
            printf("ERROR: Reached EOF on file %s", train_lbl_fn);
            exit(1);
        }
        MAT_ELE(train, i, img_w*img_h+trnlblb) = 1;
    }
    // LOADING TESTING IMAGES TO MATRIX
    for (size_t i = 0; i < n_test_img; ++i) {
        for (size_t j = 0; j < img_w*img_h; ++j) {
            if ((tstib=getc(test_img_fp)) == EOF) {
                printf("ERROR: Reached EOF on file %s", test_img_fn);
                exit(1);
            }
            MAT_ELE(test, i, j) = (float) tstib/255.f;
        }
    }
    // LOADING TESTING LABELS TO MATRIX
    for (size_t i = 0; i < n_test_lbl; ++i) {
        if ((tstlblb=getc(test_lbl_fp)) == EOF) {
            printf("ERROR: Reached EOF on file %s", test_lbl_fn);
            exit(1);
        }
        MAT_ELE(test, i, img_w*img_h+tstlblb) = 1;
    }

    printf("\n");
    printf("INFO: Loaded training images.\n");
    printf("INFO: Loaded testing images.\n");
    printf("INFO: Number of training images: %d\n", n_train_img);
    printf("INFO: Number of testing images: %d\n", n_test_img);
    printf("INFO: Image width: %d\n", img_w);
    printf("INFO: Image hight: %d\n", img_h);
    printf("INFO: Loaded training labels.\n");
    printf("INFO: Number of training labels: %d\n", n_train_lbl);
    printf("INFO: Loaded testing labels.\n");
    printf("INFO: Number of testing labels: %d\n", n_test_lbl);

    // PRINTING IMAGE
    size_t idx = 1;
    printf("\nINFO: Image index: %lu\nIMAGE: ", idx);
    for (size_t i = 0; i < img_h*img_w; ++i) {
        float p = MAT_ELE(test, idx, i)*100 - 1;
        if (p<10) {
            printf("  ");
        }
        else {
            printf("%.0f ", p);
        }
        if ((i%img_w==0)&&(i>0)) {
            printf("\n");
        }
    }
    printf("\nLABEL:\n");
    for (size_t i = 0 ; i < 10; ++i) {
        printf("%.0f -> %lu\n", MAT_ELE(test, idx, img_w*img_h+i), i);
    }

    size_t arch[] = {img_w*img_h, 16, 16, 10};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    Plot plot = {0};

    //srand(time(NULL));
    nn_rand(nn, -1, 1);

    float win_width = WIN_WIDTH_RATIO*WIN_SCALE;
    float win_hight = WIN_LENGTH_RATIO*WIN_SCALE;


    size_t batches_per_epoch = (size_t) ceilf((float)train.rows/(float)batch_s);
    //creating input vector out of in&out matrix
    Mat in_img = matrix_alloc(img_w*img_h, 1);

    for (size_t i = 0 ; i < MAX_EPOCHS; ++i) {
        float cost = 0;
        matrix_shuffle_rows(train);
        for (size_t j = 0 ; j < batches_per_epoch; ++j) {
            nn_process_batch(nn, train, RATE, j, batch_s, &cost);
        }
        printf("INFO: Epoch: %lu/%d, Cost: %.3f\n", i, MAX_EPOCHS, cost);

        in_img.ele = &MAT_ELE(test, idx, 0);
        matrix_copy(NN_INPUT(nn), in_img);
        nn_forward(nn);
        MAT_PRINT(NN_OUTPUT(nn));
    } 
    //SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    //InitWindow(win_width, win_hight, "NN");

    //SetTargetFPS(FPS);
    
    //while (!WindowShouldClose()) {
    //    int win_w = GetRenderWidth();
    //    int win_h = GetRenderHeight(); 

    //    if (IsKeyPressed(KEY_SPACE)) {
    //        paused = !paused;
    //    }

    //    if (IsKeyPressed(KEY_R)) {
    //        epoch = 0;
    //        srand(time(NULL));
    //        //nn_rand(nn, -1, 1);
    //        plot.count = 0; 
    //    }

    //    BeginDrawing();
    //    {
    //        ClearBackground((Color){0x19, 0x19, 0x19, 0xFF});
    //        DrawText("MNIST", win_w*0.02, win_h*0.03, win_w*0.05, (Color){0xE1, 0x12, 0x99, 0xFF});  

    //    }
    //    EndDrawing();
    //}
    return 0;
}
