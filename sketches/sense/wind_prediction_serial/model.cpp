#include <math.h>
#include "model.h"
#include "model_data.h"

inline float relu(float x) {
return fmaxf(0.0f, x);
}

inline float linear(float x) {
return x;
}

void predict(float* x, float* y) {

float h0[DENSE0_SIZE];

for (int i = 0; i < DENSE0_SIZE; ++i) {
h0[i] = 0.0;
for (int j = 0; j < INPUT_SIZE; ++j) {
h0[i] += (x[j] - x_mean(0)) / x_std(0) * W0(j, i);
}
h0[i] = relu(h0[i] + b0(i));
}

float h1[DENSE1_SIZE];

for (int i = 0; i < DENSE1_SIZE; ++i) {
h1[i] = 0.0;
for (int j = 0; j < DENSE0_SIZE; ++j) {
h1[i] += h0[j] * W1(j, i);
}
h1[i] = relu(h1[i] + b1(i));
}

float h2[DENSE2_SIZE];

for (int i = 0; i < DENSE2_SIZE; ++i) {
h2[i] = 0.0;
for (int j = 0; j < DENSE1_SIZE; ++j) {
h2[i] += h1[j] * W2(j, i);
}
y[i] = linear(h2[i] + b2(i));
}

}
