/*
 * Svd.cpp
 *
 *    Created on: Aug 11, 2022
 *            Author: mizolotu
 */

#include "Network.h"
#include "Layer.h"
#include "utils.h"

#include <math.h>


Network::Network() {
	y_true = new float[n_output];
}

Network::Network(short n, Layer* l[]) {

	layers = new Layer*[n];
	for (short i=0; i < n; i++) {
		layers[i] = l[i];
	}

	n_layers = n;

	n_input = layers[0]->get_input_dim();
	n_output = layers[n-1]->get_output_dim();

	y_true = new float[n_output];
}

Network::Network(short n, Layer* l, float lr, float mmntm) {

	n_layers = n;
	n_output = layers[n_layers - 1]->get_output_dim();

	learning_rate = lr;
    momentum = mmntm;

    y_true = new float[n_output];

}

Network::~Network() {}

short Network::_get_output_dim() {
    return n_output;
}

float Network::_get_outputs(short i) {
    return layers[n_layers-1]->get_outputs(i);
}

void Network::_set_inputs(float v, short i) {
	layers[0]->set_inputs(v, i);
}

void Network::_set_outputs(float v, short i) {
    y_true[i] = v;
}

void Network::train(float *x, float *y) {

	for (short i=0; i < layers[0]->get_input_dim(); i++) {
		_set_inputs(x[i], i);
	}

	for (short i=0; i < n_output; i++) {
		_set_outputs(y[i], i);
	}

	_forward();

	_backward();

}

void Network::predict(float *x) {
	for (short i=0; i < layers[0]->get_input_dim(); i++) {
		_set_inputs(x[i], i);
	}
	_forward();
}

void Network::predict(float *x, float *y) {
	for (short i=0; i < n_input; i++) {
		_set_inputs(x[i], i);
	}

	_forward();

	for (short i=0; i < n_output; i++) {
		y[i] = _get_outputs(i);
	}
}

void Network::evaluate(float *x, float *y) {
	for (short i=0; i < n_input; i++) {
		_set_inputs(x[i], i);
	}
	for (short i=0; i < n_output; i++) {
		_set_outputs(y[i], i);
	}

	_forward();
}

float Network::get_loss() {
    return loss;
}

void Network::_forward() {

    for (short i=0; i<n_layers-1; i++) {

        layers[i]->forward();

        for (short j=0; j<layers[i]->get_output_dim(); j++) {
        	layers[i + 1]->set_inputs(layers[i]->get_outputs(j), j);
        }

    }

    layers[n_layers - 1]->forward();

    loss = 0;

    for (short i=0; i<n_output; i++) {
        loss += 0.5 * pow(y_true[i] - layers[n_layers - 1]->get_outputs(i), 2);
        //std::cout << "forward output " << i << " = " << layers[n_layers - 1]->get_outputs(i) << std::endl;
    }

    //std::cout << "Loss = " << loss << std::endl;

}

void Network::_backward() {

    for (short i=0; i<n_output; i++) {
    	layers[n_layers - 1]->set_errors(i, y_true[i] - layers[n_layers - 1]->get_outputs(i));
    }

    float e;
    for (short l=n_layers-2; l>=0; l--) {

    	//std::cout << "Layer " << l << " errors:" << std::endl;

    	if (layers[l+1]->get_type() == 0) {

    		// dense

    		for (short i=0; i<layers[l]->get_output_dim(); i++) {
    			e = 0;
    			for (short j=0; j<layers[l + 1]->get_output_dim(); j++) {
    				e += layers[l + 1]->get_weights(i, j) * layers[l + 1]->get_errors(j);
    			}
    			layers[l]->set_errors(i, e);

    			//std::cout << layers[l]->get_errors(i) << ", ";
    		}

    		//std::cout << std::endl;

    	} else if (layers[l+1]->get_type() == 1) {

    		// conv1d

    		// TO DO

    	}

    }

    float grad;
    short output_i;

    for (short l=0; l<n_layers; l++) {

        // Update weights

    	if (layers[l]->get_type() == 0) {

    	   // dense

    		for (short i=0; i<layers[l]->get_input_dim(); i++) {
    			for (short j=0; j<layers[l]->get_output_dim(); j++) {

    				grad = layers[l]->get_errors(j) * layers[l]->get_d_outputs(j) * layers[l]->get_inputs(i);
    				if (grad < grad_min) {
    					grad = grad_min;
    				} else if (grad > grad_max) {
    					grad = grad_max;
    				}

    				//layers[l]->set_d_weights(learning_rate * grad + momentum * layers[l]->get_d_weights(i, j), i, j);
    				//layers[l]->set_weights(layers[l]->get_weights(i, j) + layers[l]->get_d_weights(i, j), i, j);
    				layers[l]->set_weights(layers[l]->get_weights(i, j) + learning_rate * grad, i, j);

    			}
    		}


    	} else if (layers[l]->get_type() == 1) {

    	    // conv1d

    		for (int i = 0; i < layers[l]->get_n_filters(); i++) {
    			for (int j = 0; j < layers[l]->get_kernel(); j++) {
    				for (int k = 0; k < layers[l]->get_n_channels(); k++) {
    					grad = 0;
    					output_i = 0;
    					for (int m = 0; m <= layers[l]->get_n_input() - layers[l]->get_kernel(); m += layers[l]->get_stride()) {
    						grad += layers[l]->get_errors(output_i * layers[l]->get_n_filters() + i) * layers[l]->get_d_outputs(output_i * layers[l]->get_n_filters() + i) * layers[l]->get_inputs((m + j) * layers[l]->get_n_channels() + k);
    						//std::cout << i << ", " << j << ", " << k << ", " << m << ", " << m + j << ", " << output_i << ", " << output_i * layers[l]->get_n_filters() + i << std::endl;
    						//std::cout << layers[l]->get_errors(output_i * layers[l]->get_n_filters() + i) << ", " << layers[l]->get_outputs(output_i * layers[l]->get_n_filters() + i) << ", " << layers[l]->get_d_outputs(output_i * layers[l]->get_n_filters() + i) << ", " << layers[l]->get_inputs((m + j) * layers[l]->get_n_channels() + k) << std::endl;
    						output_i += 1;
    					}
    					//std::cout << "w grad " << i << "," << j << "," << k << " = " << grad << std::endl;
    					if (grad < grad_min) {
    						grad = grad_min;
    					} else if (grad > grad_max) {
    						grad = grad_max;
    					}
    					/*if (grad != 0) {
    						std::cout << "w before = " << layers[l]->get_weights(i, j, k) << std::endl;
    					}*/

    					//layers[l]->set_d_weights(learning_rate * grad + momentum * layers[l]->get_d_weights(i, j, k), i, j, k);
    					//layers[l]->set_weights(layers[l]->get_weights(i, j, k) + layers[l]->get_d_weights(i, j, k), i, j, k);
    					layers[l]->set_weights(layers[l]->get_weights(i, j, k) + learning_rate * grad, i, j, k);

    					/*if (grad != 0) {
    						std::cout << "w after = " << layers[l]->get_weights(i, j, k) << std::endl;
    					}*/
    				}
    		    }
    		}

    	}

        // Update biases

    	if (layers[l]->get_type() == 0) {

    	    // dense

    		for (short j=0; j<layers[l]->get_output_dim(); j++) {
    			grad = layers[l]->get_errors(j) * layers[l]->get_d_outputs(j);
    			if (grad < grad_min) {
    				grad = grad_min;
    			} else if (grad > grad_max) {
    				grad = grad_max;
    			}

    			//layers[l]->set_d_biases(learning_rate * grad + momentum * layers[l]->get_d_biases(j), j);
    			//layers[l]->set_biases(layers[l]->get_biases(j) + layers[l]->get_d_biases(j), j);
    			layers[l]->set_biases(layers[l]->get_biases(j) + learning_rate * grad, j);

    		}

    	} else if (layers[l]->get_type() == 1) {

    	    // conv1d

    		for (int i = 0; i < layers[l]->get_n_filters(); i++) {
    			grad = 0;
   			    for (int j = 0; j < layers[l]->get_n_output(); j++) {
    			    grad += layers[l]->get_errors(j * layers[l]->get_n_filters() + i) * layers[l]->get_d_outputs(j * layers[l]->get_n_filters() + i);
    			}
   			   //std::cout << "b grad " << i << " = " << grad << std::endl;
    			if (grad < grad_min) {
    				grad = grad_min;
    			} else if (grad > grad_max) {
    				grad = grad_max;
    			}
    			/*if (grad != 0) {
    				std::cout << "b before = " << layers[l]->get_d_biases(i) << std::endl;
    			}*/

    			//layers[l]->set_d_biases(learning_rate * grad + momentum * layers[l]->get_d_biases(i), i);
    			//layers[l]->set_biases(layers[l]->get_biases(i) + layers[l]->get_d_biases(i), i);
    			layers[l]->set_biases(layers[l]->get_biases(i) + learning_rate * grad, i);

    			/*if (grad != 0) {
    				std::cout << "b after = " << layers[l]->get_d_biases(i) << std::endl;
    			}*/
    		}
    	}

    }
}
