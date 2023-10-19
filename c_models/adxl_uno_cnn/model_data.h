#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <avr/pgmspace.h>

extern const float xmin_data[];
extern const float xmax_data[];
extern const float W0_data[];
extern const float b0_data[];
extern const float W1_data[];
extern const float b1_data[];
extern const float W2_data[];
extern const float b2_data[];

const int INPUT_SIZE[2] = {64, 3};
const int HIDDEN0_SIZE[2] = {4, 16};
const int HIDDEN1_SIZE = 16;
const int HIDDEN2_SIZE = 3;

inline float xmin(int i) {
    return pgm_read_float_near(xmin_data + i);
}

inline float xmax(int i) {
    return pgm_read_float_near(xmax_data + i);
}

inline float W0(int i, int j, int k) {
  return pgm_read_float_near(W0_data + i * HIDDEN0_SIZE[0] * HIDDEN0_SIZE[1] + j * HIDDEN0_SIZE[1] + k);
}

inline float b0(int i) {
    return pgm_read_float_near(b0_data + i);
}

inline float W1(int i, int j) {
  return pgm_read_float_near(W1_data + i * HIDDEN1_SIZE + j);
}

inline float b1(int i) {
    return pgm_read_float_near(b1_data + i);
}

inline float W2(int i, int j) {
  return pgm_read_float_near(W2_data + i * HIDDEN2_SIZE + j);
}

inline float b2(int i) {
    return pgm_read_float_near(b2_data + i);
}

#endif

