/*
 * Skmpp.cpp
 *
 *  Created on: Jun 2, 2023
 *      Author: mizolotu
 */

#include "Skmpp.h"

#include <math.h>
#include "utils.h"

//#include <iostream>

Skmpp::Skmpp() {
	batch_probs = new float[batch_size];
	for (short i=0; i < batch_size; i++) {
		batch_probs[i] = 0.0;
	}
	centroid_indexes = new int[n_clusters];
	for (short i=0; i < n_clusters; i++) {
		centroid_indexes[i] = 0;
	}
	batch = new float[batch_size * input_dim];
	for (short i=0; i < batch_size; i++) {
		for (short j=0; j < input_dim; j++) {
			batch[i * input_dim + j] = 0.0;
		}
	}
	centroids = new float[n_clusters * input_dim];
	for (short i=0; i < n_clusters; i++) {
		for (short j=0; j < input_dim; j++) {
			centroids[i * input_dim + j] = 0.0;
		}
	}
	centroids_new = new float[n_clusters * input_dim];
	for (short i=0; i < n_clusters; i++) {
		for (short j=0; j < input_dim; j++) {
			centroids_new[i * input_dim + j] = 0.0;
		}
	}
	centroid_weights = new float[n_clusters];
	for (short i=0; i < n_clusters; i++) {
		centroid_weights[i] = 0.0;
	}

	score_n = new int[n_clusters];
	score_sum = new float[n_clusters];
	score_ssum = new float[n_clusters];
	score_max = new float[n_clusters];
	score_thr = new float[n_clusters];
	for (short i=0; i < n_clusters; i++) {
		score_n[i] = 0;
		score_sum[i] = 0.0;
		score_ssum[i] = 0.0;
		score_max[i] = 0.0;
		score_thr[i] = 0.0;
	}

	prototype_probs = new float[n_clusters];
	for (short i=0; i < n_clusters; i++) {
		prototype_probs[i] = 0.0;
	}
	prototype_indexes = new int[n_clusters];
	for (short i=0; i < n_clusters; i++) {
		prototype_indexes[i] = 0;
	}
	prototypes = new float[2 * n_clusters * input_dim];
	for (short i=0; i < 2 * n_clusters; i++) {
		for (short j=0; j < input_dim; j++) {
			prototypes[i * input_dim + j] = 0.0;
		}
	}
	prototype_weights = new float[2 * n_clusters];
	for (short i=0; i < 2 * n_clusters; i++) {
		prototype_weights[i] = 0.0;
	}
	prototype_indexes_sorted = new int[2 * n_clusters];
	for (short i=0; i < 2 * n_clusters; i++) {
		prototype_indexes_sorted[i] = i;
	}
}

Skmpp::~Skmpp() {
	// TODO Auto-generated destructor stub
}

void Skmpp::predict(float *x) {
	if (!is_score_thr_frozen) {
		_freeze_score_thr();
	}
	_cluster_point(x);
}

void Skmpp::validate(float *x) {
	if (is_score_thr_frozen) {
		_unfreeze_score_thr();
	}
	_cluster_point(x);
	if (n_val < n_max) {
		n_val++;
	}
}

void Skmpp::train(float *x) {

	// add the new vector to the batch

	for (short i=0; i < input_dim; i++) {
		batch[feature_vector_count * input_dim + i] = x[i];
	}

	/*
	std::cout << "batch:" << std::endl;
	for(short i = 0; i < batch_size; i++){
		for(short j = 0; j < input_dim; j++){
			std::cout << batch[i * input_dim + j] << ',';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	*/

	feature_vector_count += 1;

	if (feature_vector_count == batch_size) {
		is_trainable = true;
	}

	// if there are batch_size vectors in the batch, do one training iteration

	if ((is_trainable) && feature_vector_count % batch_size == 0) {

	    if (n_centroids == 0) {

	    	for(short i = 0; i < batch_size; i++) {
	            if (i == 0){
	            	batch_probs[i] = 1.0 / batch_size;
	            } else {
	            	batch_probs[i] = batch_probs[i-1] + 1.0 / batch_size;
	            }
	          }

	          _random_indexes_without_repetition(batch_size, n_clusters, centroid_indexes, batch_probs);

	          // assign centroids

	          for(short i = 0; i < n_clusters; i++){
	        	  for(short j = 0; j < input_dim; j++){
	        		  centroids[i * input_dim + j] = batch[centroid_indexes[i] * input_dim + j];
	        	  }
	          }

	          // assign weights

	          float d = 0.0;
	          int prototype_dist_argmin = 0;

	          for(short i = 0; i < batch_size; i++){
	        	  batch_probs[i] = pow(10, 8);
	        	  for(short j = 0; j < n_clusters; j++){
	        		  d = 0.0;
	        		  for(short k = 0; k < input_dim; k++){
	        			  d += pow(batch[i * input_dim + k] - centroids[j * input_dim + k], 2);
	        		  }
	        		  if (d < batch_probs[i]){
	        			  batch_probs[i] = d;
	        			  prototype_dist_argmin = j;
	        		  }
	        	  }
	        	  centroid_weights[prototype_dist_argmin] += 1;
	          }

	          /*
	          for(short i = 0; i < n_clusters; i++){
	        	  std::cout << "centroid " << i <<" = ";
	        	  for(short j = 0; j < input_dim; j++){
	        		  std::cout << centroids[i * input_dim + j] << ',';
	        	  }
	        	  std::cout << std::endl;
	        	  std::cout << "weight " << i << " = " << centroid_weights[i] << std::endl;

	          }
	          */


	          n_centroids += n_clusters;

	    } else {

	          float cost = 0.0;
	          float d = 0.0;
	          float d_min = 0.0;
	          int d_argmin = 0;

	          // find and assign prototypes

	          for(short i = 0; i < batch_size; i++){
	        	  batch_probs[i] = pow(10, 8);
	        	  for(short j = 0; j < n_clusters; j++){
	        		  for(short k = 0; k < input_dim; k++){
	        			  d += pow(batch[i * input_dim + k] - centroids[j * input_dim + k], 2);
	        		  }
	        		  if (d < batch_probs[i]){
	        			  batch_probs[i] = d;
	        		  }
	        	  }
	        	  cost += batch_probs[i];
	          }
	          cost = cost * (1.0 + 0.001 * batch_size);
	          batch_probs[0] = (batch_probs[0] + 0.001 * cost) / cost;
	          for(short i = 1; i < batch_size; i++){
	        	  batch_probs[i] = batch_probs[i-1] + (batch_probs[i] + 0.001 * cost) / cost;
	          }

	          _random_indexes_without_repetition(batch_size, n_clusters, prototype_indexes, batch_probs);

	          for(short i = 0; i < n_clusters; i++){
	        	  for(short j = 0; j < input_dim; j++){
	        		  prototypes[i * input_dim + j] = centroids[i * input_dim + j];
	        	  }
	          }

	          for(short i = 0; i < n_clusters; i++){
	        	  for(short j = 0; j < input_dim; j++){
	        		  prototypes[(n_clusters + i) * input_dim + j] = batch[prototype_indexes[i] * input_dim + j];
	        	  }
	          }

	          // assign prototype weights

	          d = 0.0;
	          d_min = 0;
	          d_argmin = 0;

	          for(short i = 0; i < n_clusters; i++){
	        	  prototype_weights[i] = centroid_weights[i];
	          }

	          for(short i = 0; i < n_clusters; i++){
	        	  prototype_weights[n_clusters + i] = 0.0;
	          }

	          for(short i = 0; i < batch_size; i++){
	        	  batch_probs[i] = pow(10, 8);
	        	  for(short j = 0; j < n_clusters; j++){
	        		  d = 0.0;
	        		  for(short k = 0; k < input_dim; k++){
	        			  d += pow(batch[i * input_dim + k] - prototypes[(n_clusters + j) * input_dim + k], 2);
	        		  }
	        		  if (d < batch_probs[i]){
	        			  batch_probs[i] = d;
	        			  d_argmin = n_clusters + j;
	        		  }
	        	  }
	        	  prototype_weights[d_argmin] += 1;
	          }

	          // weighted k-means

	          int a;
	          for (int i = 0; i < 2 * n_clusters; i++) {
	        	  for (int j = i + 1; j < 2 * n_clusters; j++){
	        		  //if (prototype_weights[prototype_indexes_sorted[i]] < prototype_weights[prototype_indexes_sorted[j]]) {
	        		  if (prototype_weights[prototype_indexes_sorted[i]] == 0) {
	        			  a = prototype_indexes_sorted[i];
	        			  prototype_indexes_sorted[i] = prototype_indexes_sorted[j];
	        			  prototype_indexes_sorted[j] = a;
	        		  }
	        	  }
	          }

	          for(short i = 0; i < n_clusters; i++){
	        	  for(short k = 0; k < input_dim; k++){
	        		  centroids[i * input_dim + k] = prototypes[prototype_indexes_sorted[i] * input_dim + k];
	        	  }
	          }

	          /*
	          std::cout << "Before k-means:" << std::endl;
	          for(short i = 0; i < 2 * n_clusters; i++){
	        	  std::cout << "proto " << i << " = ";
	        	  for(short j = 0; j < input_dim; j++){
	        		  std::cout << prototypes[prototype_indexes_sorted[i] * input_dim + j] << ',';
	        	  }
	        	  std::cout << std::endl;
	        	  std::cout << "weight " << i << " = " << prototype_weights[prototype_indexes_sorted[i]] << std::endl;
	          }
	          for(short i = 0; i < n_clusters; i++){
	        	  std::cout << "centroid " << i << " = ";
	        	  for(short j = 0; j < input_dim; j++){
	        		  std::cout << centroids[i * input_dim + j] << ',';
	        	  }
	        	  std::cout << std::endl;
	        	  std::cout << "weight " << i << " = " << prototype_weights[prototype_indexes_sorted[i]] << std::endl;
	          }
	          */

	          bool have_centroids_changed;

	          for(short iter = 0; iter < n_kmeans_iters; iter++){

	        	  for(short i = 0; i < n_clusters; i++){
	        		  centroid_weights[i] = 0.0;
	        		  for(short k = 0; k < input_dim; k++){
	        			  centroids_new[i * input_dim + k] = 0.0;
	        		  }
	        	  }

	        	  for(short i = 0; i < 2 * n_clusters; i++){
	        		  d_min = pow(10, 8);
	        		  if (prototype_weights[i] > 0) {
	        			  for(short j = 0; j < n_clusters; j++){
	        				  d = 0.0;
	        				  for(short k = 0; k < input_dim; k++){
	        					  d += pow(prototypes[i * input_dim + k] - centroids[j * input_dim + k], 2);
	        				  }
	        				  if (d < d_min){
	        					  d_min = d;
	        					  d_argmin = j;
	        				  }
	        			  }
	        			  for(short k = 0; k < input_dim; k++){
	        				  centroids_new[d_argmin * input_dim + k] += prototype_weights[i] * prototypes[i * input_dim + k];
	        			  }
	        			  centroid_weights[d_argmin] += prototype_weights[i];
	        		  }
	        	  }

	        	  for(short i = 0; i < n_clusters; i++){
	        		  for(short k = 0; k < input_dim; k++){
	        			  centroids_new[i * input_dim + k] /= centroid_weights[i];
	        		  }
	        	  }

	        	  have_centroids_changed = false;

	        	  for(short i = 0; i < n_clusters; i++){
	        		  if (have_centroids_changed == true) {
	        			  break;
	        		  }
	        		  for(short k = 0; k < input_dim; k++){
	        			  if (centroids_new[i * input_dim + k] != centroids[i * input_dim + k]) {
	        				  have_centroids_changed = true;
	        				  break;
	        			  }
	        		  }
	        	  }

	        	  //std::cout << "Have centroids changed: " << have_centroids_changed << std::endl;

	        	  if (have_centroids_changed == false) {
	        		  break;
	        	  }

	        	  for(short i = 0; i < n_clusters; i++){
	        		  for(short k = 0; k < input_dim; k++){
	        			  centroids[i * input_dim + k] = centroids_new[i * input_dim + k];
	        		  }
	        	  }

	          }

	          for(short i = 0; i < n_clusters; i++){
	        	  for(short k = 0; k < input_dim; k++){
	        		  centroids[i * input_dim + k] = centroids_new[i * input_dim + k];
	        	  }
	        	  centroid_weights[i] = 0.0;
	          }

	          // assign centroid weights

	          for(short i = 0; i < 2 * n_clusters; i++){
	        	  d_min = pow(10, 8);
	        	  for(short j = 0; j < n_clusters; j++){
	        		  d = 0.0;
	        		  for(short k = 0; k < input_dim; k++){
	        			  d += pow(prototypes[i * input_dim + k] - centroids[j * input_dim + k], 2);
	        		  }
	        		  if (d < d_min){
	        			  d_min = d;
	        			  d_argmin = j;
	        		  }
	        	  }
	        	  centroid_weights[d_argmin] += prototype_weights[i];
	          }

	          /*
	          std::cout << "After k-means:" << std::endl;
	          for(short i = 0; i < n_clusters; i++){
	        	  std::cout << "centroid " << i << " = ";
	        	  for(short j = 0; j < input_dim; j++){
	        		  std::cout << centroids[i * input_dim + j] << ',';
	        	  }
	        	  std::cout << std::endl;
	        	  std::cout << "weight " << i << " = " << centroid_weights[i] << std::endl;
	          }
	          */

	    }

	    feature_vector_count = 0;
	}

	if (n_train < n_max) {
		n_train++;
	}

}

void Skmpp::_freeze_score_thr() {
	for(short j = 0; j < n_clusters; j++){
		if (score_n[j] > 0) {
			score_thr[j] = score_alpha1 * score_sum[j] / score_n[j] + score_alpha2 * sqrt((maximum(0.0, score_ssum[j] - score_n[j] * pow(score_sum[j] / score_n[j], 2))) / score_n[j]);
		} else {
			score_thr[j] = 0.0;
		}
	}

	is_score_thr_frozen = true;
}

void Skmpp::_unfreeze_score_thr() {
	for (short i=0; i < n_clusters; i++) {
		score_n[i] = 0;
		score_sum[i] = 0.0;
		score_ssum[i] = 0.0;
		score_max[i] = 0.0;
		score_thr[i] = 0.0;
	}
    is_score_thr_frozen = false;
}

float Skmpp::get_score_thr() {
	return score_thr[centroid_idx];
}

int Skmpp::get_score_n(int i) {
    return score_n[i];
}

float Skmpp::get_score_sum(int i) {
    return score_sum[i];
}

float Skmpp::get_score_ssum(int i) {
    return score_ssum[i];
}

int Skmpp::get_n_clusters() {
    return n_clusters;
}

void Skmpp::_cluster_point(float *x) {

	float d = 0.0;
	float d_min = pow(10, 8);
	int d_argmin = 0;

	for(short j = 0; j < n_clusters; j++) {
		d = 0.0;
		for(short k = 0; k < input_dim; k++) {
			d += pow(x[k] - centroids[j * input_dim + k], 2);
		}
		if (d < d_min){
			d_min = d;
			d_argmin = j;
		}
	}

	loss = d_min;
	centroid_idx = d_argmin;

	score = pow(loss, 0.5);

	if (!is_score_thr_frozen) {
		score_n[centroid_idx] += 1;
		score_sum[centroid_idx] += score;
		score_ssum[centroid_idx] += pow(score, 2);
		if (score > score_max[centroid_idx]) {
			score_max[centroid_idx] = score;
	    }
	}
}

void Skmpp::_random_indexes_without_repetition(int imax, int n, int *indexes, float *cumsum_probs) {

	bool duplicate_found;
	float rnd;

	for(int i = 0; i < n; i++){

		duplicate_found = true;

		while(duplicate_found == true){

			rnd = (rand() % 1000)/1000.0;

			for (short j=0; j < imax; j++){
				if (rnd < cumsum_probs[j]){
					indexes[i] = j;
					break;
				}
			}

			duplicate_found = false;

			for(int j = 0; j < i; j++){
				if(indexes[i] == indexes[j]){
					duplicate_found = true;
					break;
				}
			}
		}
	}

	for(short i = 0; i < n; i++){
		for(short j = i+1; j < n; j++){
			if(indexes[i] > indexes[j]){
				rnd = indexes[i];
				indexes[i] = indexes[j];
				indexes[j] = rnd;
			}
		}
	}
}
