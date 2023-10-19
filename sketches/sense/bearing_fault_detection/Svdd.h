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

  short n_layers;
  Dense *layers;
  short n_output;

  float *c;
  //std::vector<float> c;

  short c_n;
  bool is_c_frozen;
  float learning_rate, momentum;
  float grad_min, grad_max;

  float loss;
  DynamicDimensionQueue loss_q;
  float loss_epsilon;
  float loss_alpha;
  float loss_beta;

  int score_n;
  float score_sum;
  float score_ssum;
  short score_alpha;
  float score_max;
  float score_thr;
  bool is_score_thr_frozen;
  float score;
  short score_q_size;
  float score_q[8];

  public:

    Svdd(short n, Dense *l, float lr, float mmntm, short a);
    virtual ~Svdd();

    void set_inputs(float v, short i);
    short get_output_dim();
    float get_outputs(short i);
    //void forward(float* x);
    void forward();
    void freeze_c();
    void freeze_c(float c_manual[]);
    void unfreeze_c();
      //void backward(float* x);
    void backward();
      float get_c(short i);
      float get_loss();
      float get_score();
      void freeze_score_thr();
      void unfreeze_score_thr();
      float get_score_thr();
      void clear_score_q();
      float calculate_priority_w();
};

#endif /* SVD_H_ */
