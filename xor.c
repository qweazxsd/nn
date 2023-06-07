#include <assert.h>
#include <float.h>
#include <math.h>
#include <raylib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#define NN_IMPLEMENTATION
#define NN_GYM
#include "nn.h"
#include "rlgl.h"
#include "External/stb_image.h"
#include "External/stb_image_write.h"
//---------------------------------------------------
//---------------- HYPER PARAMETERS -----------------
//---------------------------------------------------
size_t arch[] = {2, 2, 1};

#define FPS 60
#define WIN_WIDTH_RATIO 16
#define WIN_LENGTH_RATIO 9
#define WIN_SCALE 100
#define MAX_EPOCHS 10*1000
#define EPOCHS_PER_FRAME 100 
#define RATE 1.f
size_t chosen_input = 1;
int paused = false;
//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

int main() {

    assert((chosen_input>=0)&&(chosen_input<arch[0]));
    
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    Plot plot = {0};

    srand(time(NULL));
    nn_rand(nn, -1, 1);

    float tai[] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    };
    float tao[] = {
        0, 
        1, 
        1, 
        0, 
    };

    Mat ti = matrix_alloc(4 , 2);
    ti.ele = tai;
    Mat to = matrix_alloc(4 , 1);
    to.ele = tao;

    float win_width = WIN_WIDTH_RATIO*WIN_SCALE;
    float win_length = WIN_LENGTH_RATIO*WIN_SCALE;
    
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(win_width, win_length, "NN");

    SetTargetFPS(FPS);
    
    size_t epoch = 0;
    while (!WindowShouldClose()) {
        int win_w = GetRenderWidth();
        int win_l = GetRenderHeight(); 

        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }

        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            srand(time(NULL));
            nn_rand(nn, -1, 1);
            plot.count = 0; 
        }
        
        if (IsKeyPressed(KEY_P)) {
            //TakeScreenshot("output.png");
            //unsigned char *pixels = rlReadScreenPixels(win_w, win_l);
            //Image screenshot = LoadImageFromScreen();
            //stbi_write_png("output.png", win_w, win_l, 4, pixels, win_w*4*sizeof(*pixels));
        }

        for (size_t i = 0 ; i < EPOCHS_PER_FRAME && epoch<MAX_EPOCHS && paused; ++i) {
            nn_backprop(nn, ti, to, RATE);
            da_append(&plot, nn_cost(nn, ti, to));
            epoch +=1;
        }

        int font_s = (int) (win_w*0.01);
        char buf[256];

        BeginDrawing();
        {

            ClearBackground((Color){0x19, 0x19, 0x19, 0xFF});

            DrawText("XOR Gate", win_w*0.05, win_l*0.07, font_s*6, (Color){0xE1, 0x12, 0x99, 0xFF});  
            matrix_copy(NN_INPUT(nn), matrix_row(ti, chosen_input));
            nn_forward(nn);
            
            //render the NN
            Box nn_box = box_init(win_w*0.35, win_l*0.35, win_w*0.57, win_l*0.08, 0);
            size_t selected_neuron[2];
            nn_render(nn, ti, chosen_input, nn_box, selected_neuron);

            //after we know the selected_neuron we can calculate the activations of due to previous neurons
            Image neuron_a = GenImageColor(1 , ti.rows, BLACK);
            for (size_t i = 0 ; i < ti.rows; ++i) {
                matrix_copy(NN_INPUT(nn), matrix_row(ti, i));
                nn_forward(nn);
                unsigned char x = 255 * MAT_ELE(nn.as[selected_neuron[0]], selected_neuron[1], 0);  //DANGAROUS: only works if the activation is in the range [0,1] like if the activation func is sigmoid
                printf("%u\n", x);
                ImageDrawPixel(&neuron_a, 0 , i, (Color){x,x,x,0xFF});
            }
            DrawTextureEx(LoadTextureFromImage(neuron_a),(Vector2){win_w*0.59,win_l*0.59}, 0 , win_w*0.045, WHITE);
            
            //render the cost plot
            Box cost_plot_box = box_init(win_w*0.35, win_l*0.5, win_w*0.05, win_l*0.3, 0);
            cost_plot_render(plot, epoch, MAX_EPOCHS, cost_plot_box);

            //the output of each NN is rendered differently
            Box out_box = box_init(win_w*0.2, win_l*0.45, win_w*0.65, win_l*0.5, 0);

            font_s = (int) (out_box.w*0.2);
            float text_xpad = (out_box.w-font_s)/2;
            DrawText("Output", out_box.xpad +text_xpad-0.3*font_s, out_box.ypad, font_s, RAYWHITE);

            font_s = (int) (out_box.w*0.17);
            for (size_t i = 0 ; i < ti.rows; ++i) {
                matrix_copy(NN_INPUT(nn), matrix_row(ti, i));
                nn_forward(nn);
                snprintf(buf, sizeof(buf), "%.0f    %.0f  ->  %.3f", MAT_ELE(ti, i, 0), MAT_ELE(ti, i, 1), MAT_ELE(NN_OUTPUT(nn), 0, 0));
                DrawText(buf, out_box.xpad, out_box.ypad+1.5*font_s*(i+1), font_s, RAYWHITE);
                
            }
        }
        EndDrawing();
    }

    CloseWindow();

	return 0;
}
