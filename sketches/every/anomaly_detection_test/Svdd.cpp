/*
 * Svd.cpp
 *
 *    Created on: Aug 11, 2022
 *            Author: mizolotu
 */

#include <math.h>

#include "Dense.h"
#include "Svdd.h"
#include "utils.h"

//#include <iostream>


Svdd::Svdd() {
	c = new float[n_output];
    for (short i=0; i < n_output; i++) {
        c[i] = 0.0;
    }
}

Svdd::Svdd(short n, Dense* l, float lr) {

	n_layers = n;
	n_output = layers[n_layers - 1].get_output_dim();

	c = new float[n_output];
    for (short i=0; i < n_output; i++) {
        c[i] = 0.0;
    }
    learning_rate = lr;
    grad_min = -0.1;
    grad_max = 0.1;
    loss = 0.0;
}

Svdd::~Svdd() {}


short Svdd::get_output_dim() {
    return n_output;
}

float Svdd::get_outputs(short i) {
    return layers[n_layers-1].get_outputs(i);
}

void Svdd::set_inputs(float v, short i) {
    layers[0].set_inputs(v, i);
}

void Svdd::train(float *x) {
	for (short i=0; i < layers[0].get_input_dim(); i++) {
		set_inputs(x[i], i);
	}
	forward();
	backward();
}

void Svdd::validate() {
	// do something;
}

void Svdd::predict(float *x) {
	for (short i=0; i < layers[0].get_input_dim(); i++) {
		set_inputs(x[i], i);
	}
	forward();
}

float Svdd::get_loss() {
    return loss;
}

void Svdd::forward() {
    for (short i=0; i<n_layers; i++) {
        layers[i].forward();
        if (i < n_layers - 1) {
            for (short j=0; j<layers[i].get_output_dim(); j++) {
                layers[i + 1].set_inputs(layers[i].get_outputs(j), j);
            }
        }
    }
    layers[n_layers - 1].forward();
    loss = 0;
    for (short i=0; i<n_output; i++) {
        loss += 0.5 * pow(c[i] - layers[n_layers - 1].get_outputs(i), 2);
    }
}

void Svdd::backward() {
	for (short i=0; i<n_output; i++) {
		layers[n_layers - 1].set_errors(i, c[i] - layers[n_layers - 1].get_outputs(i));
	}

    float e;
    for (short l=n_layers-2; l>=0; l--) {
    	for (short i=0; i<layers[l].get_output_dim(); i++) {
    		e = 0;
    		for (short j=0; j<layers[l + 1].get_output_dim(); j++) {
    			e += layers[l + 1].get_weights(i, j) * layers[l + 1].get_errors(j);
    		}
    		layers[l].set_errors(i, e);
    	}
    }

    float grad;

    for (short l=0; l<n_layers; l++) {

    	// Update weights

        for (short i=0; i<layers[l].get_input_dim(); i++) {
        	for (short j=0; j<layers[l].get_output_dim(); j++) {
        		grad = layers[l].get_errors(j) * layers[l].get_d_outputs(j) * layers[l].get_inputs(i);
                if (grad < grad_min) {
                	grad = grad_min;
                } else if (grad > grad_max) {
                	grad = grad_max;
                }
                layers[l].set_weights(layers[l].get_weights(i, j) + learning_rate * grad, i, j);
            }
        }

        // Update biases

        for (short j=0; j<layers[l].get_output_dim(); j++) {
        	grad = layers[l].get_errors(j) * layers[l].get_d_outputs(j);
        	if (grad < grad_min) {
        		grad = grad_min;
        	} else if (grad > grad_max) {
        		grad = grad_max;
        	}
        	layers[l].set_biases(layers[l].get_biases(j) + learning_rate * grad, j);
        }

    }
}
