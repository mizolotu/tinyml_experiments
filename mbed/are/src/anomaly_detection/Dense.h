/*
 * Dense.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#ifndef DENSE_H_
#define DENSE_H_

class Dense {

	short input_dim;
	short n_units;
	float (*activation)(float);
	float (*d_activation)(float);

	float *inputs;
	float *weights;
	float *d_weights;
	float *biases;
	float *d_biases;
	float *outputs;
	float *errors;

	private:

	public:

		Dense();
		Dense(short input_dim, short n_units, float (*f)(float), float (*df)(float));
    	virtual ~Dense();

    	short get_input_dim();
    	short get_output_dim();
    	float get_inputs(short i);
    	void set_inputs(float v, short i);
    	void forward();
    	void set_errors(short i, float e);
    	float get_weights(short i, short j);
    	void set_weights(float w, short i, short j);
    	float get_d_weights(short i, short j);
    	void set_d_weights(float w, short i, short j);
    	float get_biases(short i);
    	float get_d_biases(short i);
    	void set_biases(float b, short i);
    	void set_d_biases(float b, short i);
    	float get_outputs(short i);
    	float get_errors(short i);
    	float get_d_outputs(short i);

};

float linear(float x);
float relu(float x);
float sigmoid(float x);
float tanh_(float x);
float d_linear(float x);
float d_relu(float x);
float d_sigmoid(float x);
float d_tanh_(float x);

#endif /* DENSE_H_ */
