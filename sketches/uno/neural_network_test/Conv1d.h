/*
 * Dense.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Layer.h"

#ifndef CONV1D_H_
#define CONV1D_H_

class Conv1d : public Layer {

	short input_dim, n_channels, n_input;
	short output_dim, n_output;
	short n_filters, kernel, stride;
	float (*activation)(float);
	float (*d_activation)(float);

	float *inputs;
	float *weights;
	//float *d_weights;
	float *biases;
	//float *d_biases;
	float *outputs;
	float *errors;

	short type = 1;

	private:

	public:

		Conv1d();
		Conv1d(short input_dim, short n_channels, short n_filters, short kernel, short stride, float (*f)(float), float (*df)(float));
    	virtual ~Conv1d();

    	short get_type();

    	short get_input_dim();
    	short get_output_dim();

    	short get_n_input();
    	short get_n_channels();
    	short get_n_filters();
    	short get_kernel();
    	short get_stride();
    	short get_n_output();

    	float get_inputs(short i);
    	void set_inputs(float v, short i);

    	float get_errors(short i);
    	void set_errors(short i, float e);

    	float get_weights(short i, short j, short k);
    	void set_weights(float w, short i, short j, short k);

    	//float get_d_weights(short i, short j, short k);
    	//void set_d_weights(float w, short i, short j, short k);

    	float get_biases(short i);
    	void set_biases(float b, short i);

    	//float get_d_biases(short i);
    	//void set_d_biases(float b, short i);

    	float get_outputs(short i);
    	float get_d_outputs(short i);

    	void forward();

};

#endif /* CONV1D_H_ */
