/*
 * Svd.cpp
 *
 *    Created on: Aug 11, 2022
 *            Author: mizolotu
 */

#include <math.h>

#include "DynamicDimensionQueue.h"
#include "Dense.h"
#include "Svdd.h"
#include "utils.h"

//#include <iostream>


Svdd::Svdd() {

  c = new float[n_output];
    for (short i=0; i < n_output; i++) {
        c[i] = 0.0;
    }
    loss_q = DynamicDimensionQueue(1024);
    score_q = new float[score_q_size_thr];
    for (short i=0; i<score_q_size_thr; i++) {
      score_q[i] = 0.0;
    }
    is_score_thr_frozen = true;

}

Svdd::Svdd(short n, Dense* l, float lr, float mmntm, short a) {

  n_layers = n;
  n_output = layers[n_layers - 1].get_output_dim();

  c = new float[n_output];
    for (short i=0; i < n_output; i++) {
        c[i] = 0.0;
    }
    is_c_frozen = false;
    learning_rate = lr;
    momentum = mmntm;
    grad_min = -0.1;
    grad_max = 0.1;
    loss = 0.0;
    short lqs = 1024;
    loss_q = DynamicDimensionQueue(lqs);
    loss_epsilon = 0.01;
    loss_alpha = 0.6;
    loss_beta = -0.4;
    score_n = 0;
    score_sum = 0.0;
    score_ssum = 0.0;
    score = 0.0;
    score_thr = 0.0;
    score_q = new float[score_q_size_thr];
    for (short i=0; i<score_q_size_thr; i++) {
        score_q[i] = 0.0;
    }
    is_score_thr_frozen = true;
    score_alpha2 = a;
}

Svdd::~Svdd() {}


short Svdd::get_output_dim() {
    return n_output;
}

float Svdd::get_outputs(short i) {
    return layers[n_layers-1].get_outputs(i);
}

bool Svdd::get_is_c_frozen() {
  return is_c_frozen;
}

void Svdd::freeze_c() {
  for (short i=0; i<n_output; i++) {
        c[i] /= c_step;
    }

    is_c_frozen = true;
    loss = 0;
    loss_q.clear();
}

void Svdd::freeze_c(float c_manual[]) {
    for (short i=0; i<n_output; i++) {
        c[i] = c_manual[i];
    }
    is_c_frozen = true;
    loss = 0;
    loss_q.clear();
}

void Svdd::unfreeze_c() {
    for (short i=0; i<n_output; i++) {
        c[i] = 0.0;
    }
    is_c_frozen = false;
}

float Svdd::get_c(short i) {
    return c[i];
}


float Svdd::get_loss() {
    return loss;
}

float Svdd::get_score() {
    return score;
}

void Svdd::freeze_score_thr() {
    score_thr = score_alpha1 * score_max + score_alpha2 * sqrt((maximum(0.0, score_ssum - score_n * pow(score_sum / score_n, 2))) / score_n);
    is_score_thr_frozen = true;
}

void Svdd::unfreeze_score_thr() {

    clear_score_q();

    score_n = 0;
    score_sum = 0.0;
    score_ssum = 0.0;
    score_max = 0.0;
    score_thr = 0.0;
    is_score_thr_frozen = false;
}

float Svdd::get_score_thr() {
    return score_thr;
}

void Svdd::clear_score_q() {
    for (short i=0; i<score_q_size_thr; i++) {
        score_q[i] = 0.0;
    }
    score_q_size = 0;
}

float Svdd::calculate_priority_w() {
    float w;
    w = pow(loss + loss_epsilon, loss_alpha) / loss_q.mean();
    w = pow(w, loss_beta);
    if (w < 1.0) {
        w = 1.0;
    } else if (w > 3.0) {
        w = 3.0;
    }
    return w;
}

void Svdd::set_inputs(float v, short i) {
    layers[0].set_inputs(v, i);
}

void Svdd::train(float *x) {

  for (short i=0; i < layers[0].get_input_dim(); i++) {
    set_inputs(x[i], i);
  }

  forward();

  if (is_c_frozen) {

    backward();

  } else {

      c_step++;
      for (short i=0; i<n_output; i++) {
          c[i] += layers[n_layers - 1].get_outputs(i);
      }
      if (c_step == c_steps_max) {
        freeze_c();
      }

  }

}

void Svdd::validate(float *x) {
  if (is_score_thr_frozen) {
    unfreeze_score_thr();
  }
  for (short i=0; i < layers[0].get_input_dim(); i++) {
    set_inputs(x[i], i);
  }
  forward();
}

void Svdd::predict(float *x) {
  if (!is_score_thr_frozen) {
    freeze_score_thr();
  }
  for (short i=0; i < layers[0].get_input_dim(); i++) {
    set_inputs(x[i], i);
  }
  forward();
}

void Svdd::forward() {

    for (short i=0; i<n_layers; i++) {

        layers[i].forward();

        if (i < n_layers - 1) {
            for (short j=0; j<layers[i].get_output_dim(); j++) {
                layers[i + 1].set_inputs(layers[i].get_outputs(j), j);
            }
        }

    }
    layers[n_layers - 1].forward();
    loss = 0;

    for (short i=0; i<n_output; i++) {
        loss += 0.5 * pow(c[i] - layers[n_layers - 1].get_outputs(i), 2);
    }

    float prob = pow(loss + loss_epsilon, loss_alpha);
    loss_q.enqueue(prob);

    if (score_q_size < score_q_size_thr) {
      score_q[score_q_size] = loss;
        score_q_size++;
    } else {
      for (short i=0; i<score_q_size - 1; i++) {
        score_q[i] = score_q[i + 1];
      }
      score_q[score_q_size - 1] = loss;
    }

    score = 0.0;
    for (short i=0; i<score_q_size; i++) {
        score += score_q[i];
    }
    score /= score_q_size;

    if (!is_score_thr_frozen) {
        score_n += 1;
        score_sum += score;
        score_ssum += pow(score, 2);
        if (score > score_max) {
            score_max = score;
        }

    }

}

void Svdd::backward() {

    float r = (rand() % 1000) / 1000.0;
    float r_thr;
    if (loss_q.mean() > loss_q.std()) {
        r_thr = pow(loss + loss_epsilon, loss_alpha) / (loss_q.mean() - loss_q.std());
    } else {
        r_thr = 1.0;
    }

    if (r <= r_thr) {

        for (short i=0; i<n_output; i++) {
            layers[n_layers - 1].set_errors(i, c[i] - layers[n_layers - 1].get_outputs(i));
        }

        float e;
        for (short l=n_layers-2; l>=0; l--) {
            for (short i=0; i<layers[l].get_output_dim(); i++) {
                e = 0;
                for (short j=0; j<layers[l + 1].get_output_dim(); j++) {
                    e += layers[l + 1].get_weights(i, j) * layers[l + 1].get_errors(j);
                }
                layers[l].set_errors(i, e);
            }
        }

        float w = calculate_priority_w();
        float grad;

        for (short l=0; l<n_layers; l++) {

            // Update weights

            for (short i=0; i<layers[l].get_input_dim(); i++) {
                for (short j=0; j<layers[l].get_output_dim(); j++) {
                    grad = layers[l].get_errors(j) * layers[l].get_d_outputs(j) * layers[l].get_inputs(i);
                    if (grad < grad_min) {
                        grad = grad_min;
                    } else if (grad > grad_max) {
                        grad = grad_max;
                    }
                    layers[l].set_d_weights(w * (learning_rate * grad + momentum * layers[l].get_d_weights(i, j)), i, j);
                    layers[l].set_weights(layers[l].get_weights(i, j) + layers[l].get_d_weights(i, j), i, j);
                }
            }

            // Update biases

            for (short j=0; j<layers[l].get_output_dim(); j++) {
                grad = layers[l].get_errors(j) * layers[l].get_d_outputs(j);
                if (grad < grad_min) {
                    grad = grad_min;
                } else if (grad > grad_max) {
                    grad = grad_max;
                }
                layers[l].set_d_biases(w * (learning_rate * grad + momentum * layers[l].get_d_biases(j)), j);
                layers[l].set_biases(layers[l].get_biases(j) + layers[l].get_d_biases(j), j);
            }
        }

    } else {

        // do something

    }
}
