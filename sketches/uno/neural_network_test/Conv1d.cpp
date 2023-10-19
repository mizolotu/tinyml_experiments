/*
 * Conv1d.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <math.h>

#include "Conv1d.h"
#include "utils.h"

#include "Arduino.h"

Conv1d::Conv1d() {}

Conv1d::Conv1d(short input_dim_, short n_channels_, short n_filters_, short kernel_, short stride_, float (*f)(float), float (*df)(float)) {

	input_dim = input_dim_;
	n_channels = n_channels_;
	n_filters = n_filters_;
	kernel = kernel_;
	stride = stride_;
	activation = f;
	d_activation = df;

	weights = new float[n_filters * kernel * n_channels];
	//d_weights = new float[n_filters * kernel * n_channels];
	biases = new float[n_filters];
	//d_biases = new float[n_filters];

	inputs = new float[input_dim];

	n_input = (int)(input_dim / n_channels);
	n_output = (int)((n_input - kernel) / stride) + 1;

	//std::cout << n_output << std::endl;

	output_dim = n_output * n_filters;

	outputs = new float[output_dim];
	errors = new float[output_dim];

	int seed = 42;
	srand(seed);

	for (short i=0; i < n_filters * kernel * n_channels; i++) {

		weights[i] = (rand() % 2000) / 1000.0 - 1;
		//weights[i] = 1.0;

		//d_weights[i] = 0.0;
	}

	for (short i=0; i < input_dim; i++) {
		inputs[i] = 0.0;
	}


	for (short i=0; i < n_filters; i++) {

		biases[i] = (rand() % 2000) / 1000.0 - 1;
		//biases[i] = 1.0;

		//d_biases[i] = 0.0;
	}

	for (short i=0; i < output_dim; i++) {
		outputs[i] = 0.0;
		errors[i] = 0.0;
	}

	//std::cout << "Conv layer init complete!" << std::endl;

}

Conv1d::~Conv1d() {
	delete[] weights;
	//delete[] d_weights;
	delete[] biases;
	//delete[] d_biases;
}

short Conv1d::get_type() {
	return type;
}

float Conv1d::get_weights(short i, short j, short k) {
	return weights[i * kernel * n_channels + j * n_channels + k];
}

/*float Conv1d::get_d_weights(short i, short j, short k) {
	return d_weights[i * kernel * n_channels + j * n_channels + k];
}*/

void Conv1d::set_weights(float w, short i, short j, short k) {
	weights[i * kernel * n_channels + j * n_channels + k] = w;
}

/*void Conv1d::set_d_weights(float w, short i, short j, short k) {
	d_weights[i * kernel * n_channels + j * n_channels + k] = w;
}*/

float Conv1d::get_biases(short i) {
	return biases[i];
}

/*float Conv1d::get_d_biases(short i) {
	return d_biases[i];
}*/

void Conv1d::set_biases(float b, short i) {
	biases[i] = b;
}

/*void Conv1d::set_d_biases(float b, short i) {
	d_biases[i] = b;
}*/

short Conv1d::get_input_dim() {
	return input_dim;
}

float Conv1d::get_inputs(short i) {
	return inputs[i];
}

void Conv1d::set_inputs(float v, short i) {
	inputs[i] = v;
}

short Conv1d::get_output_dim() {
	return output_dim;
}

float Conv1d::get_outputs(short i) {
	return outputs[i];
}

float Conv1d::get_d_outputs(short i) {
	return d_activation(outputs[i]);
}

short Conv1d::get_n_input() {
	return n_input;
}

short Conv1d::get_n_channels() {
	return n_channels;
}

short Conv1d::get_n_filters() {
	return n_filters;
}

short Conv1d::get_kernel() {
	return kernel;
}

short Conv1d::get_stride() {
	return stride;
}

short Conv1d::get_n_output() {
	return n_output;
}

void Conv1d::forward() {

	//std::cout << "Conv1d::forward() starts!" << std::endl;

	short output_i = 0;

	for (int i = 0; i <= n_input - kernel; i += stride) {
		for (int j = 0; j < n_filters; j++) {
			outputs[output_i * n_filters + j] = 0;
			for (int k = 0; k < kernel; k++) {
				for (int m = 0; m < n_channels; m++) {
					outputs[output_i * n_filters + j] += weights[j * kernel * n_channels + k * n_channels + m] * inputs[(i + k) * n_channels + m];
					//std::cout << i << ", " << j << ", " << k << ", " << m << ", " << i + k << ", " << output_i << ", " << output_i * n_filters + j << std::endl;
					//std::cout << weights[j * kernel * n_channels + k * n_channels + m] << ", " << inputs[(i + k) * n_channels + m] << std::endl;
				}
			}
			outputs[output_i * n_filters + j] = activation(outputs[output_i * n_filters + j] + biases[j]);
			//std::cout << "Conv output " << output_i * n_filters + j << " = " << outputs[output_i * n_filters + j] << std::endl;
	    }
		output_i += 1;
	}

	/*std::cout << "conv output = ";
	for (int i = 0; i < n_output; i++) {
		for (int j = 0; j < n_filters; j++) {
			std::cout << outputs[i * n_filters + j] << ", ";
		}
		std::cout << std::endl;
	}*/

	//std::cout << "Conv1d::forward() ends!" << std::endl;

}

void Conv1d::set_errors(short i, float e) {
	errors[i] = e;
}

float Conv1d::get_errors(short i) {
	return errors[i];
}
