/*
 * Dense.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#ifndef LAYER_H_
#define LAYER_H_

class Layer {

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

		Layer();
		virtual ~Layer();

		virtual short get_type();

    	virtual short get_input_dim();
    	virtual short get_output_dim();

    	virtual float get_weights(short i, short j);
    	virtual float get_weights(short i, short j, short k);
    	virtual void set_weights(float w, short i, short j);
    	virtual void set_weights(float w, short i, short j, short k);

    	//virtual float get_d_weights(short i, short j);
    	//virtual float get_d_weights(short i, short j, short k);
    	//virtual void set_d_weights(float w, short i, short j);
    	//virtual void set_d_weights(float w, short i, short j, short k);

    	virtual void forward();

    	virtual float get_inputs(short i);
    	virtual void set_inputs(float v, short i);

    	virtual float get_errors(short i);
    	virtual void set_errors(short i, float e);

    	virtual float get_biases(short i);
    	virtual void set_biases(float b, short i);

    	//virtual float get_d_biases(short i);
    	//virtual void set_d_biases(float b, short i);

    	virtual float get_outputs(short i);
    	virtual float get_d_outputs(short i);

    	virtual short get_n_input();
    	virtual short get_n_channels();
    	virtual short get_n_filters();
    	virtual short get_kernel();
    	virtual short get_stride();
    	virtual short get_n_output();

};

#endif /* LAYER_H_ */
