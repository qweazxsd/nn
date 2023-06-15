#include <stdbool.h>
#include <math.h>
#include <raylib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define NN_IMPLEMENTATION
#include "nn.h"

//---------------------------------------------------
//---------------- HYPER PARAMETERS -----------------
//---------------------------------------------------
FILE *test_img_fp, *test_lbl_fp, *train_img_fp, *train_lbl_fp;
char test_img_fn[] = "data/test_img";
char test_lbl_fn[] = "data/test_lbl";
char train_img_fn[] = "data/train_img";
char train_lbl_fn[] = "data/train_lbl";
size_t idx = 1;


#define FPS 30
#define WIN_WIDTH_RATIO 16
#define WIN_LENGTH_RATIO 9
#define BATCHES_PER_FRAME 20
#define WIN_SCALE 100
#define MAX_EPOCHS 100
#define EPOCHS_PER_FRAME 1 
#define RATE 1e-1 
size_t batch_s = 10;
int paused = false;
//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

Vector2 nn_get_neuron_ctr(NN nn, Box b, size_t n_neu, size_t current_layer, size_t current_neu) {
    Vector2 neu_ctr;
    float neuron_distx = b.w/(nn.n_layers);
    if (n_neu==1) {
        neu_ctr.x = b.xpad+current_layer*neuron_distx;
        neu_ctr.y = b.ypad+b.l/2;
    }
    else {
        neu_ctr.x = b.xpad+current_layer*neuron_distx;
        neu_ctr.y = b.ypad+current_neu*b.l/(n_neu-1);
    }
    return neu_ctr;
}

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

    fclose(train_img_fp);
    fclose(train_lbl_fp);
    fclose(test_img_fp);
    fclose(test_lbl_fp);

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
    Mat *da = nn_act_alloc(nn);
    Plot plot = {0};
    Plot plot_acc = {0};
    Batch batch = {0};

    srand(time(NULL));
    nn_rand(nn, -1, 1);

    size_t batches_per_epoch = (size_t) ceilf((float)train.rows/(float)batch_s);
    //creating input vector out of in&out matrix
    Mat in_img = matrix_alloc(img_w*img_h, 1);

    //for (size_t i = 0 ; i < MAX_EPOCHS; ++i) {
    //    float cost = 0;
    //    matrix_shuffle_rows(train);
    //    for (size_t j = 0 ; j < batches_per_epoch; ++j) {
    //        nn_process_batch(nn, da, train, RATE, j, batch_s, &cost);
    //    }

    //    //CALCULATING ACCURACY

    //    size_t n_correct_lbl = 0;
    //    for (size_t j = 0 ; j < test.rows; ++j) {
    //        in_img.ele = &MAT_ELE(test, j, 0);
    //        matrix_copy(NN_INPUT(nn), in_img);
    //        nn_forward(nn);
    //        size_t n_correct_decision = 0;
    //        for (size_t k = 0 ; k < NN_OUTPUT(nn).rows; ++k) {
    //            bool decision = (MAT_ELE(NN_OUTPUT(nn), k, 0) >= 0.5f);
    //            if (decision == MAT_ELE(test, j, img_w*img_h+k)) {
    //                n_correct_decision += 1;
    //            }
    //        }
    //        if (n_correct_decision == 10 ) {
    //            n_correct_lbl += 1;
    //        }
    //    }
    //    printf("INFO: Epoch: %lu/%d, Cost: %.3f, Acc.: %.2f%%\n", i+1, MAX_EPOCHS, cost, 100.f*(float)n_correct_lbl/(float)test.rows);

    //    in_img.ele = &MAT_ELE(test, idx, 0);
    //    matrix_copy(NN_INPUT(nn), in_img);
    //    nn_forward(nn);
	//    for (size_t i = 0; i < NN_OUTPUT(nn).rows; i++) {
    //        printf("        ");
	//	    for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++) {
	//		    printf("%f -> %lu", MAT_ELE(NN_OUTPUT(nn), i, j), i);				
	//	    }

	//	    printf("\n");
	//    }
    //    printf("\n");
    //} 

    float win_width = WIN_WIDTH_RATIO*WIN_SCALE;
    float win_hight = WIN_LENGTH_RATIO*WIN_SCALE;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(win_width, win_hight, "NN");

    SetTargetFPS(FPS);
    
    size_t epoch = 0;

    //DRAW SELECTED IMAGE
    Image input_img = GenImageColor(img_w , img_h, BLACK);
    for (size_t i = 0 ; i < img_h; ++i) {
        for (size_t j = 0 ; j < img_w; ++j) {
            unsigned char x = 255 * MAT_ELE(test, idx, i*img_w+j);  //DANGAROUS: only works if the activation is in the range [0,1] like if the activation func is sigmoid
            ImageDrawPixel(&input_img, j , i, (Color){x,x,x,0xFF});
        }
    }
    Texture2D input_img_texture = LoadTextureFromImage(input_img); 

    while (!WindowShouldClose()) {
        int win_w = GetRenderWidth();
        int win_h = GetRenderHeight(); 

        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
            printf("INFO: Paused!");
        }

        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            srand(time(NULL));
            nn_rand(nn, -1, 1);
            plot.count = 0; 
        }

        //for (size_t i = 0 ; i < EPOCHS_PER_FRAME && epoch<MAX_EPOCHS && paused; ++i) {
        //    float cost = 0;
        //    matrix_shuffle_rows(train);
        //    for (size_t j = 0 ; j < batches_per_epoch; ++j) {
        //        nn_process_batch(nn, da, train, RATE, j, batch_s, &cost);
        //    }
        //    da_append(&plot, cost);
        //    epoch +=1;
        //}
        
        for (size_t i = 0 ; i < BATCHES_PER_FRAME && epoch<MAX_EPOCHS && paused; ++i) {
            nn_process_batch_per_frame(nn, da, train, RATE, &batch, batch_s);
            if (i%10==0) {
                da_append(&plot, batch.cost);
                size_t n_correct_lbl = 0;
                for (size_t j = 0 ; j < test.rows; ++j) {
                    in_img.ele = &MAT_ELE(test, j, 0);
                    matrix_copy(NN_INPUT(nn), in_img);
                    nn_forward(nn);
                    size_t n_correct_decision = 0;
                    for (size_t k = 0 ; k < NN_OUTPUT(nn).rows; ++k) {
                        bool decision = (MAT_ELE(NN_OUTPUT(nn), k, 0) >= 0.5f);
                        if (decision == MAT_ELE(test, j, img_w*img_h+k)) {
                            n_correct_decision += 1;
                        }
                    }
                    if (n_correct_decision == 10 ) {
                        n_correct_lbl += 1;
                    }
                } 
                da_append(&plot_acc, (float)n_correct_lbl/(float)test.rows);
            }
            if (batch.finished) {
                batch.finished= false;
                batch.start = 0;
                epoch +=1;
            }
            in_img.ele = &MAT_ELE(test, idx, 0);
            matrix_copy(NN_INPUT(nn), in_img);
            nn_forward(nn);
	        for (size_t i = 0; i < NN_OUTPUT(nn).rows; i++) {
                printf("        ");
	            for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++) {
	                printf("%f -> %lu", MAT_ELE(NN_OUTPUT(nn), i, j), i);				
	            }
	            printf("\n");
	        }
            printf("\n");

        }
        

        BeginDrawing();
        {
            ClearBackground((Color){0x19, 0x19, 0x19, 0xFF});
            DrawText("MNIST", win_w*0.02, win_h*0.03, win_w*0.05, (Color){0xE1, 0x12, 0x99, 0xFF});  

            DrawTextureEx(input_img_texture, (Vector2){win_w*0.45,win_h*0.45}, 0 , win_w*0.005, WHITE);

            in_img.ele = &MAT_ELE(test, idx, 0);
            matrix_copy(NN_INPUT(nn), in_img);
            nn_forward(nn);

            //render the cost plot
            Box cost_plot_box = box_init(win_w*0.35, win_h*0.5, win_w*0.05, win_h*0.3, 0);
            cost_plot_render(plot, epoch, MAX_EPOCHS, cost_plot_box);
            acc_plot_render(plot_acc, cost_plot_box);
            //render the NN
            Box box = box_init(win_w*0.5, win_h*0.3, win_w*0.45, win_h*0.1, 0);
            size_t selected_neuron[2];
            float line_thick = box.l*0.002;

            float neuron_distx = box.w/(nn.n_layers);
            Vector2 neuron_center;
            Vector2 pneuron_center;
            Color neuron_color_low = {0x00, 0x00, 0x00, 0xFF};
            Color neuron_color_high = {0xFF, 0xFF, 0xFF, 0xFF};
            Color w_color_low = {0xFD, 0x53, 0x53, 0xFF};
            Color w_color_high = {0x36, 0xDE, 0x7c, 0xFF};
            float circ_r;
            size_t n, np;
            char buf[256];
            int font_s;
            Color ring_color;

            //find closest neuron to the mouse
            float min_d2 = FLT_MAX;
            float d2;
            for (size_t l = 0 ; l < nn.n_layers+1; ++l) {
                n = nn.as[l].rows;
                circ_r = box.l/n*0.2;
                for (size_t i = 0; i < n; ++i) {
                    neuron_center = nn_get_neuron_ctr(nn, box, n, l, i);
                    
                    //then calculate distance between the neuron and the mouse
                    d2 = sqrtf((neuron_center.x-(float)GetMouseX())*(neuron_center.x-(float)GetMouseX())+(neuron_center.y-(float)GetMouseY())*(neuron_center.y-(float)GetMouseY()));
                    if ((l>0)&&(d2<min_d2)&&IsMouseButtonPressed(MOUSE_BUTTON_LEFT)&&(d2<circ_r)) {
                        min_d2 = d2;
                        selected_neuron[0] = l;
                        selected_neuron[1] = i;
                    }
                }
            }

            //start with rendering the weights to hide them behind the neurons
            for (size_t l = 1 ; l < nn.n_layers+1; ++l) {
                n = nn.as[l].rows;
                for (size_t i = 0; i < n; ++i) {
                    neuron_center = nn_get_neuron_ctr(nn, box, n, l, i);
                    np = nn.as[l-1].rows;
                    for (size_t j = 0 ; j < np; ++j) {
                        w_color_high.a = floorf(255.f*sigmoid(MAT_ELE(nn.ws[l-1], i, j)));
                        pneuron_center = nn_get_neuron_ctr(nn, box, np, l-1, j);
                        DrawLineEx(neuron_center, pneuron_center, line_thick, ColorAlphaBlend(w_color_low, w_color_high, WHITE));
                    }
                }
            }

            for (size_t l = 0 ; l < nn.n_layers+1; ++l) {
                n = nn.as[l].rows;
                circ_r = box.l/(sqrtf(n))*0.1;
                font_s = (circ_r*0.6);
                for (size_t i = 0; i < n; ++i) {
                    neuron_center = nn_get_neuron_ctr(nn, box, n, l, i);
                    neuron_color_high.a = floorf(255.f*MAT_ELE(nn.as[l], i, 0));
                    if (l>0 ) {
                        if ((l==selected_neuron[0])&&(i==selected_neuron[1])) {
                            ring_color = GetColor(0xEA4646FF);
                        }
                        else {
                            ring_color = (Color){0x90, 0x90, 0x90, 0xFF};
                        }
                        DrawCircleV(neuron_center, circ_r, ColorAlphaBlend(neuron_color_low, neuron_color_high, WHITE));
                        DrawRing(neuron_center, circ_r, 1.07*circ_r, 0 , 360, 1 , ring_color);
                        if (l == nn.n_layers) {
                            snprintf(buf, sizeof(buf), "%.2f", MAT_ELE(nn.as[l], i, 0));
                            DrawText(buf, neuron_center.x - 1.7*font_s/2, neuron_center.y - font_s/2, font_s, RED);
                        }
                    }
                    else {
                        ring_color = (Color){0x90, 0x90, 0x90, 0xFF};
                        DrawCircleV(neuron_center, circ_r, ColorAlphaBlend(neuron_color_low, neuron_color_high, WHITE));
                        DrawRing(neuron_center, circ_r, 1.07*circ_r, 0 , 360, 1 , ring_color);
                        //snprintf(buf, sizeof(buf), "%.2f", MAT_ELE(train, idx, i));
                        //DrawText(buf, neuron_center.x - 1.7*font_s/2, neuron_center.y - font_s/2, font_s, RED);
                    }
                }
            }

        }
        EndDrawing();
    }
    return 0;
}
