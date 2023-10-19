/*
 * Svd.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "DynamicDimensionQueue.h"
#include "Dense.h"
//#include <vector>

#ifndef SVD_H_
#define SVD_H_

class Svdd {

  short n_layers = 3;
  short n_output = 8;
  Dense layers[3] = {
    Dense(12, 32, &relu, &d_relu),
    Dense(32, 16, &relu, &d_relu),
    Dense(16, n_output, &linear, &d_linear)
  };

  float *c;

  short c_step = 0;
  bool is_c_frozen = false;
  short c_steps_max = 1024;

  float learning_rate = 0.001;
  float momentum = 0.01;
  float grad_min = -0.1;
  float grad_max = 0.1;

  float loss = 0;
  DynamicDimensionQueue loss_q;
  float loss_epsilon = 0.001;
  float loss_alpha = 0.6;
  float loss_beta = -0.4;

  int score_n = 0;
  float score_sum = 0.0;
  float score_ssum = 0.0;
  float score_alpha1 = 1.25;
  float score_alpha2 = 3.0;
  float score_max;
  float score_thr = 0.0;
  bool is_score_thr_frozen;
  float score = 0.0;
  short score_q_size_thr = 8;
  float *score_q;
  short score_q_size = 0;

  long n_train = 0.0;
  long n_val = 0.0;
  long n_max = 2147483647;

  public:

    Svdd();
    Svdd(short n, Dense *l, float lr, float mmntm, short a);
    virtual ~Svdd();

    void set_inputs(float v, short i);
    short get_output_dim();
    float get_outputs(short i);
    void forward();
    bool get_is_c_frozen();
    void freeze_c();
    void freeze_c(float c_manual[]);
    void unfreeze_c();
      void backward();
      float get_c(short i);
      float get_loss();
      float get_score();
      void freeze_score_thr();
      void unfreeze_score_thr();
      float get_score_thr();
      void clear_score_q();
      float calculate_priority_w();

      void train(float *x);
      void validate(float *x);
      void predict(float *x);

      long get_n_train();
      long get_n_val();

};

#endif /* SVD_H_ */
