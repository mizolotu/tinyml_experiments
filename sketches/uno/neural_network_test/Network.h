/*
 * Svd.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Layer.h"
#include "Dense.h"
#include "utils.h"

//#include <vector>

#ifndef NETWORK_H_
#define NETWORK_H_

class Network {

	short n_layers;

	short n_input;
	short n_output;

	float *y_true;

	float learning_rate = 0.001;
	float momentum = 0.05;
	float grad_min = -1.0;
	float grad_max = 1.0;

	float loss = 0.0;

	Layer** layers;

	public:

		Network();
		Network(short n, Layer* l[]);
		Network(short n, Layer *l, float lr, float mmntm);
		virtual ~Network();

		void train(float *x, float *y);
		void predict(float *x);
		void predict(float *x, float *y);
		void evaluate(float *x, float *y);
		float get_loss();

	private:

		void _set_inputs(float v, short i);
		void _set_outputs(float v, short i);
		short _get_output_dim();
		float _get_outputs(short i);
		void _forward();
		void _backward();
};

#endif /* NETWORK_H_ */
