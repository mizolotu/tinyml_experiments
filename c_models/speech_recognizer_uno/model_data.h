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

const int INPUT_SIZE = 84;//231;
const int DENSE0_SIZE = 16;//64;
const int DENSE1_SIZE = 16;//64;
const int DENSE2_SIZE = 3;

inline float xmin(int i) {
    return pgm_read_float_near(xmin_data + i);
}

inline float xmax(int i) {
    return pgm_read_float_near(xmax_data + i);
}

inline float W0(int i, int j) {
  return pgm_read_float_near(W0_data + i * DENSE0_SIZE + j);
}

inline float b0(int i) {
    return pgm_read_float_near(b0_data + i);
}

inline float W1(int i, int j) {
  return pgm_read_float_near(W1_data + i * DENSE1_SIZE + j);
}

inline float b1(int i) {
    return pgm_read_float_near(b1_data + i);
}

inline float W2(int i, int j) {
  return pgm_read_float_near(W2_data + i * DENSE2_SIZE + j);
}

inline float b2(int i) {
    return pgm_read_float_near(b2_data + i);
}

#endif

