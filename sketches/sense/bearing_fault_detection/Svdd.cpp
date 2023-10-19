/*
 * Svd.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <math.h>

#include "DynamicDimensionQueue.h"
#include "Dense.h"
#include "Svdd.h"
#include "utils.h"

//#include <iostream>
//using namespace std;

Svdd::Svdd(short n, Dense* l, float lr, float mmntm, short a) {
  n_layers = n;
  layers = l;
  n_output = layers[n_layers - 1].get_output_dim();

  //
  c = new float[n_output];
  //

  for (short i=0; i < n_output; i++) {

    c[i] = 0.0;
    //c.push_back(0.0);

  }
  is_c_frozen = false;
  c_n = 0;
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
  for (short i=0; i<8; i++) {
    score_q[i] = 0.0;
  }
  score_q_size = 0;
  is_score_thr_frozen = true;
  score_alpha = a;
}


Svdd::~Svdd() {}


short Svdd::get_output_dim() {
  return n_output;
}

float Svdd::get_outputs(short i) {
  return layers[n_layers-1].get_outputs(i);
}

void Svdd::freeze_c() {
  //cout << "c_n = " << c_n << endl;
  for (short i=0; i<n_output; i++) {
    //cout << c[i] << ",";
    c[i] /= c_n;
  }
  //cout << endl;layers[l].get_errors(j) * layers[l].get_d_outputs(j) * layers[l].get_inputs(i)
  is_c_frozen = true;
  loss_q.clear();
}

void Svdd::freeze_c(float c_manual[]) {
  for (short i=0; i<n_output; i++) {
    c[i] = c_manual[i];
  }
  is_c_frozen = true;
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
  //cout << score_sum << endl;
  //cout << score_ssum << endl;
  //cout << score_n << endl;
  //score_thr = score_sum / score_n + score_alpha * sqrt((maximum(0.0, score_ssum - score_n * pow(score_sum / score_n, 2))) / score_n);
  score_thr = score_max + score_alpha * sqrt((maximum(0.0, score_ssum - score_n * pow(score_sum / score_n, 2))) / score_n);
  is_score_thr_frozen = true;
}

void Svdd::unfreeze_score_thr() {

  //score_q.clear();

  //for (short i=0; i<8; i++) {
  //  score_q[i] = 0.0;
  //}
  //score_q_size = 0;

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

  //score_q.clear();

  for (short i=0; i<8; i++) {
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

//void Svdd::forward(float* x) {
void Svdd::forward() {
  //float* layer_output;
  //layer_output = x;

  for (short i=0; i<n_layers; i++) {

    /*
    cout << "input of layer " << i + 1 << " out of " << n_layers << endl;
    for (short j=0; j<layers[i].get_input_dim(); j++) {
      cout << layers[i].get_inputs(j) << ",";
    }
    cout << endl;
    */

    //layers[i].forward(layer_output);
    layers[i].forward();

    //layer_output = layers[i].get_outputs();

    if (i < n_layers - 1) {
      //cout << "output of layer " << i + 1 << " out of " << n_layers << endl;
      for (short j=0; j<layers[i].get_output_dim(); j++) {
        //cout << layers[i].get_outputs(j) << ",";
        layers[i + 1].set_inputs(layers[i].get_outputs(j), j);
      }
      //cout << endl;
    }
  }
  layers[n_layers - 1].forward();

  /*
  cout << "output of layer " << n_layers << " out of " << n_layers << endl;
  for (short j=0; j<layers[n_layers - 1].get_output_dim(); j++) {
    cout << layers[n_layers - 1].get_outputs(j) << ",";
  }
  cout << endl;
  */

  loss = 0;
  for (short i=0; i<n_output; i++) {
    loss += 0.5 * pow(c[i] - layers[n_layers - 1].get_outputs(i), 2);
  }

  float prob = pow(loss + loss_epsilon, loss_alpha);
  //cout << "Enqueed: " << prob << endl;

  loss_q.enqueue(prob);

  /*
  int s = loss_q.size();
  float xmx = loss_q.xmax();
  cout << "Queue size: " << s << ", " << "Queue max: " << xmx << endl;
  */

  //score_q.enqueue(loss);

  if (score_q_size < 8) {
    score_q_size ++;
  }

  for (short i=0; i<score_q_size - 1; i++) {
    score_q[i] = score_q[i + 1];
  }
  score_q[score_q_size - 1] = loss;


  //score = score_q.mean();
  score = 0.0;
  for (short i=0; i<score_q_size; i++) {
    score += score_q[i];
  }
  score /= score_q_size;

  if (is_c_frozen) {

    // do something here or nah?

  } else {

    c_n += 1;
    for (short i=0; i<n_output; i++) {
      //cout << layers[n_layers - 1].get_outputs(i) << ",";
      c[i] += layers[n_layers - 1].get_outputs(i);
    }
    //cout << endl;

  }

  if (is_score_thr_frozen) {

    // do something here or nah?

  } else {

    score_n += 1;
    score_sum += score;
    score_ssum += pow(score, 2);
    if (score > score_max) {
      score_max = score;
    }

  }

}

//void Svdd::backward(float* x) {
void Svdd::backward() {

  float r = (rand() % 1000) / 1000.0;
  float r_thr;
  if (loss_q.mean() > loss_q.std()) {
    r_thr = pow(loss + loss_epsilon, loss_alpha) / (loss_q.mean() - loss_q.std());
  } else {
    r_thr = 1.0;
  }

  /*
  cout << "loss q max: " << loss_q.xmax() << endl;
  cout << "loss q mean: " << loss_q.mean() << endl;
  cout << "loss: " << loss << endl;
  cout << "r: " << r << endl;
  cout << "r_thr: " << r_thr << endl;
  */

  if (r <= r_thr) {

    //cout << "TRAINING" << endl;

    // Last layer errors

    for (short i=0; i<n_output; i++) {
      layers[n_layers - 1].set_errors(i, c[i] - layers[n_layers - 1].get_outputs(i));
    }

    // Other layer errors

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

    /*
    cout << "Loss: " << loss << endl;
    cout << "W: " << w << endl;
     */

    // Update weights of the first layer

    /*
    for (short i=0; i<layers[0].get_input_dim(); i++) {
      for (short j=0; j<layers[0].get_output_dim(); j++) {
        layers[0].set_d_weights(w * (learning_rate * layers[0].get_errors(j) * layers[0].get_d_outputs(j) * x[i] + momentum * layers[0].get_d_weights(i, j)), i, j);
            layers[0].set_weights(layers[0].get_weights(i, j) + layers[0].get_d_weights(i, j), i, j);
        layers[0].set_weights(layers[0].get_weights(i, j) + learning_rate * layers[0].get_errors(j) * layers[0].get_d_outputs(j) * x[i], i, j);
      }
      }

    // Update biases of the first layer

    for (short j=0; j<layers[0].get_output_dim(); j++) {
      layers[0].set_d_biases(w * (learning_rate * layers[0].get_errors(j) * layers[0].get_d_outputs(j) + momentum * layers[0].get_d_biases(j)), j);
      layers[0].set_biases(layers[0].get_biases(j) + layers[0].get_d_biases(j), j);
      layers[0].set_biases(layers[0].get_biases(j) + learning_rate * layers[0].get_errors(j) * layers[0].get_d_outputs(j), j);
      }

    // Update the rest of the layers

     */

    //for (short l=1; l<n_layers; l++) {

    float grad;

    for (short l=0; l<n_layers; l++) {

      // Update weights

      for (short i=0; i<layers[l].get_input_dim(); i++) {
        for (short j=0; j<layers[l].get_output_dim(); j++) {
          //layers[l].set_d_weights(w * (learning_rate * layers[l].get_errors(j) * layers[l].get_d_outputs(j) * layers[l-1].get_outputs(i) + momentum * layers[l].get_d_weights(i, j)), i, j);
          grad = layers[l].get_errors(j) * layers[l].get_d_outputs(j) * layers[l].get_inputs(i);
          if (grad < grad_min) {
            grad = grad_min;
          } else if (grad > grad_max) {
            grad = grad_max;
          }
          layers[l].set_d_weights(w * (learning_rate * grad + momentum * layers[l].get_d_weights(i, j)), i, j);
          layers[l].set_weights(layers[l].get_weights(i, j) + layers[l].get_d_weights(i, j), i, j);
          //layers[l].set_weights(layers[l].get_weights(i, j) + learning_rate * layers[l].get_errors(j) * layers[l].get_d_outputs(j) * layers[l-1].get_outputs(i), i, j);
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
        //layers[l].set_biases(layers[l].get_biases(j) + learning_rate * layers[l].get_errors(j) * layers[l].get_d_outputs(j), j);
      }
    }

  } else {

    // do something

  }
}
