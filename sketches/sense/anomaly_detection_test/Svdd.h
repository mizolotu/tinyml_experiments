/*
 * Svd.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Dense.h"
//#include <vector>

#ifndef SVD_H_
#define SVD_H_

class Svdd {

	short n_layers = 2;
  short n_hidden = 110;
  short n_output = 85;
  Dense layers[2] = {
    Dense(12, n_hidden, &relu, &d_relu),
    Dense(n_hidden, n_output, &linear, &d_linear)
  };

	float *c;

	float learning_rate = 0.001;
	float momentum = 0.01;
	float grad_min = -0.1;
	float grad_max = 0.1;

	float loss = 0;

	public:

		Svdd();
		Svdd(short n, Dense *l, float lr);
		virtual ~Svdd();

		void set_inputs(float v, short i);
		short get_output_dim();
		float get_outputs(short i);
		void forward();
		void backward();
		void train(float *x);
		void validate();
    	void predict(float *x);
    	float get_loss();

};

#endif /* SVD_H_ */
