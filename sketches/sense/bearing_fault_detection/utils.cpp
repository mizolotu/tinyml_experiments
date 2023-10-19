/*
 * utils.cpp
 *
 *  Created on: Aug 23, 2022
 *      Author: mizolotu
 */

#include <math.h>

#include "utils.h"

float linear(float x) {
    return x;
}

float relu(float x) {
    return fmaxf(0.0f, x);
}

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float tanh_(float x) {
  return tanh(x);
}

float d_linear(float x) {
    return 1.0;
}

float d_relu(float x) {
  float d;
  if (x > 0) {
    d = 1;
  } else {
    d = 0;
  }
    return d;
}

float d_sigmoid(float x) {
  return x * (1.0 - x);
}

float d_tanh_(float x) {
  return 1.0 - pow(tanh(x), 2);
}

float maximum(float x, float y){
  if (x >= y) {
    return x;
  } else {
    return y;
  }
}
