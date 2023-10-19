/*
 * Dense.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <math.h>

#include "Layer.h"
#include "utils.h"

Layer::Layer() {}

Layer::~Layer() {}

short Layer::get_type() {
	return 0;
}

float Layer::get_weights(short i, short j) {
	return 0.0;
}

float Layer::get_weights(short i, short j, short k) {
	return 0.0;
}

/*float Layer::get_d_weights(short i, short j) {
	return 0.0;
}*/

/*float Layer::get_d_weights(short i, short j, short k) {
	return 0.0;
}*/

void Layer::set_weights(float w, short i, short j) {}

void Layer::set_weights(float w, short i, short j, short k) {}

//void Layer::set_d_weights(float w, short i, short j) {}

//void Layer::set_d_weights(float w, short i, short j, short k) {}

float Layer::get_biases(short i) {
	return biases[i];
}

/*float Layer::get_d_biases(short i) {
	return d_biases[i];
}*/

void Layer::set_biases(float b, short i) {
	biases[i] = b;
}

/*void Layer::set_d_biases(float b, short i) {
	d_biases[i] = b;
}*/

short Layer::get_input_dim() {
	return 0.0;
}

float Layer::get_inputs(short i) {
	return inputs[i];
}

void Layer::set_inputs(float v, short i) {
	inputs[i] = v;
}

short Layer::get_output_dim() {
	return 0.0;
}

float Layer::get_outputs(short i) {
	return outputs[i];
}

float Layer::get_d_outputs(short i) {
	return d_activation(outputs[i]);
}

void Layer::forward() {}

void Layer::set_errors(short i, float e) {
	errors[i] = e;
}

float Layer::get_errors(short i) {
	return errors[i];
}

short Layer::get_n_input() {
	return 0.0;
}

short Layer::get_n_channels() {
	return 0.0;
}

short Layer::get_n_filters() {
	return 0.0;
}

short Layer::get_kernel() {
	return 0.0;
}

short Layer::get_stride() {
	return 0.0;
}

short Layer::get_n_output() {
	return 0.0;
}
