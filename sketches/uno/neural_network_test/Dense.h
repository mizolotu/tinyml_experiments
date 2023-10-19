/*
 * Dense.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Layer.h"

#ifndef DENSE_H_
#define DENSE_H_

class Dense : public Layer {

	short input_dim;
	short n_units;
	float (*activation)(float);
	float (*d_activation)(float);

	float *inputs;
	float *weights;
	//float *d_weights;
	float *biases;
	//float *d_biases;
	float *outputs;
	float *errors;

	short type = 0;

	private:

	public:

		Dense();
		Dense(short input_dim, short n_units, float (*f)(float), float (*df)(float));
    	virtual ~Dense();

    	short get_type();

    	short get_input_dim();
    	short get_output_dim();

    	float get_inputs(short i);
    	void set_inputs(float v, short i);

    	float get_errors(short i);
    	void set_errors(short i, float e);

    	float get_weights(short i, short j);
    	void set_weights(float w, short i, short j);

    	//float get_d_weights(short i, short j);
    	//void set_d_weights(float w, short i, short j);

    	float get_biases(short i);
    	void set_biases(float b, short i);

    	//float get_d_biases(short i);
    	//void set_d_biases(float b, short i);

    	float get_outputs(short i);
    	float get_d_outputs(short i);

    	void forward();

};

#endif /* DENSE_H_ */
