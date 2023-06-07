#ifndef NN_H_

#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <raylib.h>
#include <float.h>


#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])


typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *ele;
} Mat;

#define MAT_ELE(m, i, j) (m).ele[(i)*(m).stride + (j)]

Mat matrix_alloc(size_t rows, size_t cols);
#define VEC_ALLOC(i) matrix_alloc((i), 1)  //allocate row vector
Mat matrix_eye_alloc(size_t n);
Mat matrix_row(Mat m, size_t row);
void matrix_populate(Mat m, float *a, size_t len_a);  //DANGAROUS: NO CHECK FOR ARRAY SHAPE, JUST TOTAL NUMBER OF ELEMENTS
void matrix_copy(Mat dst, Mat src);
void matrix_fill(Mat m, float v);
void matrix_shuffle_rows(Mat m);
void matrix_rand(Mat m, float low, float high);
void matrix_mul (Mat dst, Mat a, Mat b);
void matrix_add(Mat a, Mat b);
void matrix_print(Mat m, const char *name);
#define MAT_PRINT(m) matrix_print(m, #m);
void matrix_act(Mat m);

typedef struct {
    size_t n_layers;
    Mat *ws;  // these are arrays of matrices
    Mat *bs;
    Mat *as;  // number of activations is 1 more than the number of layers
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).n_layers]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
void nn_print(NN nn);
void nn_backprop(NN nn, Mat ti, Mat to, float rate, size_t cost_div);
float nn_sum_cost(NN nn, Mat ti, Mat to);

typedef struct {
    float *items;
    size_t count;
    size_t capacity;
} Plot;

#define DA_INIT_CAP 256
#define da_append(da, item)                                                          \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;   \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL);                                             \
        }                                                                            \
                                                                                     \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)

typedef struct {
    float l;
    float w;
    float xpad;
    float ypad;
} Box;


Box box_init(float w, float l, float xoffset, float yoffset, int draw_box);
void nn_render(NN nn, Mat ti, size_t chosen_input, Box box, size_t *selected_neuron);

#endif // NN_H_ 

#ifdef NN_IMPLEMENTATION

int rand_int(int low, int high) {
    return low + rand()%(high-low+1);
}

float rand_float(float low, float high) {
    float scale = (float) ( (double)rand()/(double)RAND_MAX );
    return low + scale*(high-low);
}

float sigmoid(float x) {
    return 1.f/(1.f+expf(-x));
}

float sigmoid_prime(float x) { //as a function of the activation itself
    return x*(1 - x);
}

#ifndef ACT 
#define ACT sigmoid
#endif
#ifndef ACT_PRIME 
#define ACT_PRIME sigmoid_prime
#endif

Mat matrix_alloc(size_t rows, size_t cols) {
	
	Mat m;
	m.rows = rows;
	m.cols = cols;
	m.stride = cols;
	
	m.ele = (float *) malloc(sizeof(*m.ele)*rows*cols);
	assert(m.ele != NULL);
	
	return m;
}

Mat matrix_eye_alloc(size_t n) {
	
	Mat m;
	m.rows = n;
	m.cols = n;
	m.stride = n;
	
	m.ele = (float *) malloc(sizeof(*m.ele)*n*n);
	assert(m.ele != NULL);
	
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
            if (i==j) {
			    MAT_ELE(m, i, j) = 1;
            }
		}
	}

	return m;
}

Mat matrix_row(Mat m, size_t row) {
    Mat out = VEC_ALLOC(m.cols);
    for (size_t i = 0; i < m.cols; ++i) {
        MAT_ELE(out, i, 0) = MAT_ELE(m, row, i);
    }
    return out;
}

void matrix_populate(Mat m, float *a, size_t len_a) {
    if (len_a != m.rows*m.cols) {
        printf("ERROR: Array length must be equal to total number of elements.");
        exit(1);
    }

	for (size_t i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			MAT_ELE(m, i, j) = a[i*m.stride+j];
		}
	}
}

void matrix_copy(Mat dst, Mat src) {
    assert(dst.cols==src.cols);
    assert(dst.rows==src.rows);

	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
			MAT_ELE(dst, i, j) = MAT_ELE(src, i, j);
		}
	}

}

void matrix_fill(Mat m, float v) {
	for (size_t i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			MAT_ELE(m, i, j) = v;
		}
	}
}

void matrix_shuffle_rows(Mat m) {
    for (size_t i = 0; i < m.rows; ++i) {
        //srand(time(NULL));
        size_t j = rand_int(i, m.rows-1);
        if (i!=j) {
            for (size_t k = 0; k < m.cols; ++k) {
                float temp = MAT_ELE(m, i, k);
                MAT_ELE(m, i, k) = MAT_ELE(m, j, k); 
                MAT_ELE(m, j, k) = temp; 
            }
        }
    }
}

void matrix_rand(Mat m, float low, float high) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_ELE(m, i, j) = rand_float(low, high);
        }
    }
}

void matrix_mul (Mat dst, Mat a, Mat b) {
	if (a.cols != b.rows) {
		printf("ERROR: First matrix column number must be equal to second matrix row number.");
		exit(1);
	}

	if (dst.rows != a.rows) {
		printf("ERROR: Destination matrix row number must be equal to first matrix row number.");
		exit(1);
	}

	if (dst.cols != b.cols) {
		printf("ERROR: Destination matrix column number must be equal to second matrix column number.");
		exit(1);
	}

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
                MAT_ELE(dst, i, j) = 0;  // just to make sure the element is set to zero
            for (size_t k = 0; k < a.cols; k++) {
                MAT_ELE(dst, i, j) += MAT_ELE(a, i, k) * MAT_ELE(b, k, j);
            }
        }
    }
}


void matrix_add(Mat dst, Mat a) {
	if (dst.rows != a.rows) {
		printf("ERROR: First matrix row number must be equal to second matrix row number.");
		exit(1);
	}

	if (dst.cols != a.cols) {
		printf("ERROR: First matrix col number must be equal to second matrix col number.");
		exit(1);
	}

	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
            MAT_ELE(dst, i, j) += MAT_ELE(a, i, j);
	    }
    }
}

void matrix_print(Mat m, const char *name) {
	
	printf("%s =\n", name);
	for (size_t i = 0; i < m.rows; i++) {
        printf("        ");
		for (size_t j = 0; j < m.cols; j++) {
			printf("%f ", MAT_ELE(m, i, j));				
		}

		printf("\n");
	}
    printf("\n");
}

void matrix_act(Mat m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0 ; j < m.cols; ++j) {
            MAT_ELE(m, i, j) = ACT(MAT_ELE(m, i, j));
        }     
    }
}


NN nn_alloc(size_t *arch, size_t arch_count) {
    assert(arch_count>0);

    NN nn;
    nn.n_layers = arch_count-1;

    nn.ws = malloc(sizeof(*nn.ws)*(nn.n_layers));
    assert(nn.ws!=NULL);
    nn.bs = malloc(sizeof(*nn.bs)*(nn.n_layers));
    assert(nn.bs!=NULL);
    nn.as = malloc(sizeof(*nn.as)*(nn.n_layers+1));
    assert(nn.as!=NULL);

    nn.as[0] = matrix_alloc(arch[0], 1);
    for (size_t i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = matrix_alloc(arch[i],nn.as[i-1].rows);  // number of rows is the number of rows of the output and cols is the number of rows of the input
        nn.bs[i-1] = VEC_ALLOC(arch[i]);
        nn.as[i]   = VEC_ALLOC(arch[i]);
    }
    
    return nn;
}

void nn_rand(NN nn, float low, float high) {
    for (size_t l = 0; l < nn.n_layers; ++l) {
        matrix_rand(nn.ws[l], low, high);
        matrix_rand(nn.bs[l], low, high);
    }
}

void nn_forward(NN nn) {
    for (size_t l = 0; l < nn.n_layers; ++l) {
        matrix_mul(nn.as[l+1], nn.ws[l], nn.as[l]);
        matrix_add(nn.as[l+1], nn.bs[l]);
        matrix_act(nn.as[l+1]);
    }
}

void nn_print(NN nn) {
    char buf[256];
    for (size_t i = 0; i < nn.n_layers; ++i) {
        snprintf(buf, sizeof(buf), "ws[%zu]", i);
        matrix_print(nn.ws[i], buf);
        snprintf(buf, sizeof(buf), "bs[%zu]", i);
        matrix_print(nn.bs[i], buf);
    }
}

void nn_backprop(NN nn, Mat ti, Mat to, float rate, size_t cost_div) {
    assert(ti.cols == NN_INPUT(nn).rows);
    assert(to.cols == NN_OUTPUT(nn).rows);
    assert(ti.rows == to.rows);

    size_t n = ti.rows;  //number of inputs

    //creating a duplicate of the NN activations to store dCda
    Mat *da;
    da = malloc(sizeof(*da)*(nn.n_layers+1));
    assert(da!=NULL);

    for (size_t l = 0; l < nn.n_layers+1; ++l) {
        da[l] = VEC_ALLOC(nn.as[l].rows);
    }

    for (size_t i = 0; i < n; ++i) {  // for every input

        // setting NN_INPUT as training_input
        matrix_copy(NN_INPUT(nn), matrix_row(ti, i));
        nn_forward(nn);

        for (size_t l = 0; l < nn.n_layers+1; ++l) {
            matrix_fill(da[l], 0);
        }

        //setting dCda of the last layer as the difference (NN_OUTPUT - training_output)
        for (size_t j = 0; j < NN_OUTPUT(nn).rows; ++j) {
            MAT_ELE(da[nn.n_layers], j, 0) = (MAT_ELE(NN_OUTPUT(nn), j, 0) - MAT_ELE(to, i, j))*2/n;
        }
        
        //starting backpropagation
        for (size_t l = nn.n_layers; l > 0; --l) {  //for every layer (exept the last one which is the input) starting form the last, hence backpropagation
            for (size_t j = 0; j < nn.as[l].rows; ++j) { //for the number of neorons in the current layer
                float a  = MAT_ELE(nn.as[l], j, 0);  //current activation
                float dCda = MAT_ELE(da[l], j, 0);
                MAT_ELE(nn.bs[l-1], j, 0) -= rate*dCda*ACT_PRIME(a); // DANGAROUS: not all activation functions can be expressed in terms of the activation itself, some might need to define z
                for (size_t k = 0; k < nn.as[l-1].rows; ++k) { //for the number of neorons in the previous layer
                    float w  = MAT_ELE(nn.ws[l-1], j, k);
                    float pa = MAT_ELE(nn.as[l-1], k, 0);  //previous activation
                    MAT_ELE(nn.ws[l-1], j, k) -= rate*dCda*ACT_PRIME(a)*pa;// DANGAROUS: not all activation functions can be expressed in terms of the activation itself, some might need to define z
                    MAT_ELE(da[l-1], k, 0) = dCda*ACT_PRIME(a)*w;// DANGAROUS: not all activation functions can be expressed in terms of the activation itself, some might need to define z
                }
            }
        }
    }
}

float nn_sum_cost(NN nn, Mat ti, Mat to) {
    assert(ti.cols == NN_INPUT(nn).rows);
    assert(to.cols == NN_OUTPUT(nn).rows);
    assert(ti.rows == to.rows);

    size_t n = ti.rows;  //number of inputs

    float c = 0;
    for (size_t i = 0; i < n; ++i) {  // for every input
        Mat x = matrix_row(ti, i);
        Mat y = matrix_row(to, i);
        // setting NN_INPUT as training_input
        matrix_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j) {
            float diff = MAT_ELE(NN_OUTPUT(nn), j, 0) - MAT_ELE(y, j, 0);
            c += diff*diff;
        }
    }

    return c;
}

Box box_init(float w, float l, float xoffset, float yoffset, int draw_box) {
    Box b; 
    b.l = l;
    b.w = w;
    b.xpad = xoffset;
    b.ypad = yoffset;

    if (draw_box) {
        float line_thick = b.l*0.01;
        DrawLineEx((Vector2) {b.xpad, b.ypad}, (Vector2){b.xpad+b.w, b.ypad}, line_thick,(Color){0x00, 0xFF, 0x00, 0xFF});
        DrawLineEx((Vector2) {b.xpad, b.ypad+b.l}, (Vector2){b.xpad+b.w, b.ypad+b.l}, line_thick,(Color){0x00, 0xFF, 0x00, 0xFF});
        DrawLineEx((Vector2) {b.xpad, b.ypad}, (Vector2){b.xpad, b.ypad+b.l}, line_thick,(Color){0x00, 0xFF, 0x00, 0xFF});
        DrawLineEx((Vector2) {b.xpad+b.w, b.ypad}, (Vector2){b.xpad+b.w, b.ypad+b.l}, line_thick,(Color){0x00, 0xFF, 0x00, 0xFF});
        DrawCircle(b.xpad+b.w/2 , b.ypad+b.l/2 , 5, RED); 
    }
    return b;
}

void nn_render(NN nn, Mat ti, size_t chosen_input, Box box, size_t *selected_neuron) {
    if (ARRAY_LEN(selected_neuron)>2) {
        printf("ERROR: The selected nueron has only a layer and index.\n");
        exit(1);
    }

    float line_thick = box.l*0.01;

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

    //start with rendering the weights to hide them behind the neurons
    for (size_t l = 0 ; l < nn.n_layers+1; ++l) {
        n = nn.as[l].rows;
        for (size_t i = 0; i < n; ++i) {
            if (n==1) {
                neuron_center.x = box.xpad+l*neuron_distx;
                neuron_center.y = box.ypad+box.l/2;
            }
            else {
                neuron_center.x = box.xpad+l*neuron_distx;
                neuron_center.y = box.ypad+i*box.l/(n-1);
            }

            if (l>0 ) {
                np = nn.as[l-1].rows;
                for (size_t j = 0 ; j < np; ++j) {
                    w_color_high.a = floorf(255.f*sigmoid(MAT_ELE(nn.ws[l-1], i, j)));
                    if (np==1) {
                        pneuron_center.x = box.xpad+(l-1)*neuron_distx;
                        pneuron_center.y = box.ypad+box.l/2;
                    }
                    else {
                        pneuron_center.x = box.xpad+(l-1)*neuron_distx;
                        pneuron_center.y = box.ypad+j*box.l/(np-1);
                    }

                    DrawLineEx(neuron_center, pneuron_center, line_thick, ColorAlphaBlend(w_color_low, w_color_high, WHITE));
                }
            } 
        }
    }

    //find closest neuron to the mouse
    float min_d2 = FLT_MAX;
    float d2;
    for (size_t l = 0 ; l < nn.n_layers+1; ++l) {
        n = nn.as[l].rows;
        circ_r = box.l/n*0.2;
        for (size_t i = 0; i < n; ++i) {
            //first calculate neuron center
            if (n==1) {
                neuron_center.x = box.xpad+l*neuron_distx;
                neuron_center.y = box.ypad+box.l/2;
            }
            else {
                neuron_center.x = box.xpad+l*neuron_distx;
                neuron_center.y = box.ypad+i*box.l/(n-1);
            }
            
            //then calculate distance between the neuron and the mouse
            d2 = sqrtf((neuron_center.x-(float)GetMouseX())*(neuron_center.x-(float)GetMouseX())+(neuron_center.y-(float)GetMouseY())*(neuron_center.y-(float)GetMouseY()));
            if ((l>0)&&(d2<min_d2)&&IsMouseButtonPressed(MOUSE_BUTTON_LEFT)&&(d2<circ_r)) {
                min_d2 = d2;
                selected_neuron[0] = l;
                selected_neuron[1] = i;
            }
        }
    }

            Color ring_color;
    //render the neorons
    for (size_t l = 0 ; l < nn.n_layers+1; ++l) {
        n = nn.as[l].rows;
        circ_r = box.l/n*0.2;
        font_s = (int) (circ_r*0.6);
        for (size_t i = 0; i < n; ++i) {
            neuron_color_high.a = floorf(255.f*MAT_ELE(nn.as[l], i, 0));

            //first calculate neuron center
            if (n==1) {
                neuron_center.x = box.xpad+l*neuron_distx;
                neuron_center.y = box.ypad+box.l/2;
            }
            else {
                neuron_center.x = box.xpad+l*neuron_distx;
                neuron_center.y = box.ypad+i*box.l/(n-1);
            }

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
                snprintf(buf, sizeof(buf), "%.2f", MAT_ELE(ti, chosen_input, i));
                DrawText(buf, neuron_center.x - 1.7*font_s/2, neuron_center.y - font_s/2, font_s, RED);
            }
        }
    }
} 

void cost_plot_render(Plot plot, size_t epoch, size_t max_epochs, Box b) {
    float max = FLT_MIN;
    float min = FLT_MAX;
    int font_s = (int) (b.l*0.05);
    float line_thick = b.l*0.01;
    char buf[64];

    for (size_t i = 0 ; i < plot.count; ++i) {
        if (plot.items[i]<min) min=plot.items[i];
        if (plot.items[i]>max) max=plot.items[i];
    }

    if (min>0) min=0;
        
    for (size_t i = 0 ; i+1 < plot.count; ++i) {  //+1 to not draw on the first point 
        size_t n = plot.count;
        if (n<1000) n=1000;
        float x1 = b.xpad + line_thick + b.w/n*i;
        float x2 = b.xpad + line_thick + b.w/n*(i+1);
        float y1 = b.ypad + ( 1 - (plot.items[i]-min)/(max-min) )*b.l;
        float y2 = b.ypad + ( 1 - (plot.items[i+1]-min)/(max-min) )*b.l;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, line_thick*0.7, RED);
    }
    snprintf(buf, sizeof(buf), "Epoch %lu/%lu", epoch, max_epochs);
    DrawText(buf, b.xpad+b.w-9*font_s, b.ypad+b.l+2*line_thick, font_s, RAYWHITE);
    DrawText("Cost", b.xpad-0.6*font_s, b.ypad-1.1*font_s, font_s, RAYWHITE);
    DrawLineEx((Vector2){b.xpad, b.ypad+b.l}, (Vector2){b.xpad+b.w, b.ypad+b.l}, 0.01*b.l, RAYWHITE);
    DrawLineEx((Vector2){b.xpad, b.ypad+b.l}, (Vector2){b.xpad, b.ypad}, 0.01*b.l, RAYWHITE);

}

#endif // NN_IMPLEMENTATION 

