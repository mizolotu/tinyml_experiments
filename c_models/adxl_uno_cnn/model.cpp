#include <math.h>
#include "model.h"
#include "model_data.h"

inline float relu(float x) {
    return fmaxf(0.0f, x); 
}

inline float sigmoid(float x) {
    return 1/(1+exp(-x)); 
}

float * predict(float x[INPUT_SIZE[0]][INPUT_SIZE[1]]) {

    //stride = ((INPUT_SIZE[0] - HIDDEN0_SIZE[1]) / (HIDDEN0_SIZE[0] - 1))
    float h0[HIDDEN0_SIZE[0]][HIDDEN1_SIZE[1]];

    for (int k = 0; k < HIDDEN0_SIZE; k++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
    	    h0[k][j] = 0;
            for (int i = 0; i < INPUT_SIZE; i++) {
                if (xmax(i) > xmin(i)) {
	                h0[j] += (x[i] - xmin(i)) / (xmax(i) - xmin(i)) * W0(i, j);
	            }
            }
            h0[j] = relu(h0[j] + b0(j));
        }
    }

    
    float h1[DENSE1_SIZE];

    for (int j = 0; j < DENSE1_SIZE; j++) {
     	h1[j] = 0;
        for (int i = 0; i < DENSE0_SIZE; i++) {
            h1[j] += h0[i] * W1(i, j);
        }
        h1[j] = relu(h1[j] + b1(j));
    }
    
    float h2[DENSE2_SIZE];

    for (int j = 0; j < DENSE2_SIZE; j++) {
    	h2[j] = 0;
        for (int i = 0; i < DENSE1_SIZE; i++) {
            h2[j] += h1[i] * W2(i, j);
        }
        h2[j] = h2[j] + b2(j);
    }
    
    static float y[DENSE2_SIZE];
    float s = 0;
    for (int i = 0; i < DENSE2_SIZE; i++) {
        y[i] = exp(h2[i]);
        s += y[i];
    }
    for (int i = 0; i < DENSE2_SIZE; i++) {
        y[i] = y[i] / s;
    }
        
    return y;
}
