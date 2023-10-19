/*
 * Dense.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <math.h>

#include "Dense.h"

Dense::Dense() {}

Dense::Dense(short input_dim_, short n_units_, float (*f)(float), float (*df)(float)) {

	input_dim = input_dim_;
	n_units = n_units_;
	activation = f;
	d_activation = df;

	weights = new float[input_dim * n_units];
	d_weights = new float[input_dim * n_units];
	biases = new float[n_units];
	d_biases = new float[n_units];

	inputs = new float[input_dim];
	outputs = new float[n_units];
	errors = new float[n_units];

	int seed = 42;
	srand(seed);

	for (short i=0; i < input_dim * n_units; i++) {
		weights[i] = (rand() % 2000) / 1000.0 - 1;
		d_weights[i] = 0.0;
	}

	for (short i=0; i < input_dim; i++) {
		inputs[i] = 0.0;
	}

	for (short i=0; i < n_units; i++) {
		biases[i] = (rand() % 2000) / 1000.0 - 1;
		d_biases[i] = 0.0;
		outputs[i] = 0.0;
		errors[i] = 0.0;
	}
}

Dense::~Dense() {
	delete[] weights;
	delete[] d_weights;
	delete[] biases;
	delete[] d_biases;
}

float Dense::get_weights(short i, short j) {
	return weights[i * n_units + j];
}

float Dense::get_d_weights(short i, short j) {
	return d_weights[i * n_units + j];
}

void Dense::set_weights(float w, short i, short j) {
	weights[i * n_units + j] = w;
}

void Dense::set_d_weights(float w, short i, short j) {
	d_weights[i * n_units + j] = w;
}

float Dense::get_biases(short i) {
	return biases[i];
}

float Dense::get_d_biases(short i) {
	return d_biases[i];
}

void Dense::set_biases(float b, short i) {
	biases[i] = b;
}

void Dense::set_d_biases(float b, short i) {
	d_biases[i] = b;
}

short Dense::get_input_dim() {
	return input_dim;
}

float Dense::get_inputs(short i) {
	return inputs[i];
}

void Dense::set_inputs(float v, short i) {
	inputs[i] = v;
}

short Dense::get_output_dim() {
	return n_units;
}

float Dense::get_outputs(short i) {
	return outputs[i];
}

float Dense::get_d_outputs(short i) {
	return d_activation(outputs[i]);
}

void Dense::forward() {
	for (int j = 0; j < n_units; j++) {
		outputs[j] = 0;
	    for (int i = 0; i < input_dim; i++) {
	    	outputs[j] += inputs[i] * weights[i * n_units + j];
	    }
	    outputs[j] = activation(outputs[j] + biases[j]);
	}
}

void Dense::set_errors(short i, float e) {
	errors[i] = e;
}

float Dense::get_errors(short i) {
	return errors[i];
}

float linear(float x) {
    return x;
}

float relu(float x) {
    return fmaxf(0.0f, x);
}

float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

float tanh_(float x) {
	return tanh(x);
}

float d_linear(float x) {
    return 1.0;
}

float d_relu(float x) {
	float d;
	if (x > 0) {
		d = 1;
	} else {
		d = 0;
	}
    return d;
}

float d_sigmoid(float x) {
	return x * (1.0 - x);
}

float d_tanh_(float x) {
	return 1.0 - pow(tanh(x), 2);
}
