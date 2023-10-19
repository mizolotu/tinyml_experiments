#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <avr/pgmspace.h>

extern const float x_mean_data[];
extern const float x_std_data[];
extern const float y0_mean_data[];

extern const float W0_data[];
extern const float b0_data[];
extern const float W1_data[];
extern const float b1_data[];
extern const float W2_data[];
extern const float b2_data[];

const int INPUT_SIZE = 60;

inline float x_mean(int i) {
return pgm_read_float_near(x_mean_data + i);
}

inline float x_std(int i) {
return pgm_read_float_near(x_std_data + i);
}

inline float y0_mean(int i) {
return pgm_read_float_near(y0_mean_data + i);
}

const int DENSE0_SIZE = 32;

inline float W0(int i, int j) {
return pgm_read_float_near(W0_data + i * DENSE0_SIZE + j);
}
inline float b0(int i) {
return pgm_read_float_near(b0_data + i);
}

const int DENSE1_SIZE = 32;

inline float W1(int i, int j) {
return pgm_read_float_near(W1_data + i * DENSE1_SIZE + j);
}
inline float b1(int i) {
return pgm_read_float_near(b1_data + i);
}

const int DENSE2_SIZE = 1;

inline float W2(int i, int j) {
return pgm_read_float_near(W2_data + i * DENSE2_SIZE + j);
}
inline float b2(int i) {
return pgm_read_float_near(b2_data + i);
}

#endif

