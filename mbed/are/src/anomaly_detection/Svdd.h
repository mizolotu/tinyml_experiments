/*
 * Svd.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Model.h"
#include "DynamicDimensionQueue.h"
#include "Dense.h"
//#include <vector>

#ifndef SVD_H_
#define SVD_H_

class Svdd : public Model {

	short n_layers = 2;
	short n_hidden = 36;
	short n_output = 22;
	Dense layers[2] = {
		Dense(12, n_hidden, &relu, &d_relu),
		Dense(n_hidden, n_output, &linear, &d_linear)
	};

	float *c;

	short c_step = 0;
	bool is_c_frozen = false;
	short c_steps_max = 1024;

	float learning_rate = 0.001;
	float momentum = 0.01;
	float grad_min = -0.1;
	float grad_max = 0.1;

	DynamicDimensionQueue loss_q;
	float loss_epsilon = 0.01;
	float loss_alpha = 0.6;
	float loss_beta = -0.4;

	int score_n = 0;
	float score_sum = 0.0;
	float score_ssum = 0.0;
	float score_max = 0.0;
	float score_thr = 0.0;

	public:

		Svdd();
		Svdd(short n, Dense *l, float lr, float mmntm, short a);
		virtual ~Svdd();

		void train(float *x);
		void validate(float *x);
		void predict(float *x);
		float get_score_thr();
		int get_score_n();
		float get_score_sum();
		float get_score_ssum();

	private:

		void _freeze_score_thr();
		void _unfreeze_score_thr();

		void _set_inputs(float v, short i);
		short _get_output_dim();
		float _get_outputs(short i);
		void _forward();
		bool _get_is_c_frozen();
		void _freeze_c();
		void _freeze_c(float c_manual[]);
		void _unfreeze_c();
    	void _backward();
    	float _get_c(short i);
    	float _calculate_priority_w();

};

#endif /* SVD_H_ */
