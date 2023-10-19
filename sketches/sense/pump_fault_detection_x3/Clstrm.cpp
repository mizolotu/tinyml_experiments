/*
 * Clstrm.cpp
 *
 *  Created on: Jun 6, 2023
 *      Author: mizolotu
 */

#include "Clstrm.h"

#include <math.h>
#include "utils.h"

//#include <iostream>

Clstrm::Clstrm() {

	n = new float[n_microclusters];
	microcluster_indexes_sorted = new int[n_microclusters];

	ls = new float[n_microclusters * input_dim];
	ss = new float[n_microclusters * input_dim];

	for (short i=0; i < n_microclusters; i++) {
		n[i] = 0.0;
		microcluster_indexes_sorted[i] = i;
		for (short j=0; j < input_dim; j++) {
			ls[i * input_dim + j] = 0.0;
			ss[i * input_dim + j] = 0.0;
		}
	}

	centroids = new float[n_clusters * input_dim];
	centroids_new = new float[n_clusters * input_dim];
	centroid_radii = new float[n_clusters];
	centroid_n = new float[n_clusters];
	for (short i=0; i < n_clusters; i++) {
		for (short j=0; j < input_dim; j++) {
			centroids[i * input_dim + j] = 0.0;
			centroids_new[i * input_dim + j] = 0.0;
			centroid_radii[i] = 0.0;
		}
		centroid_n[i] = 0.0;
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

}

Clstrm::~Clstrm() {
	// TODO Auto-generated destructor stub
}

void Clstrm::train(float *x) {

	if (n_centroids < 2) {

		n[n_centroids] = 1;
		for (short i=0; i < input_dim; i++) {
			ls[n_centroids * input_dim + i] = x[i];
			ss[n_centroids * input_dim + i] = pow(x[i], 2);
		}

		n_centroids++;

	} else {

		float d, d_min;
		float d_between_centroids_min = pow(10, 8);
		int d_between_centroids_argmin0, d_between_centroids_argmin1;

		for (short i=0; i < n_centroids; i++) {
			for (short j=i+1; j < n_centroids; j++) {
				d = 0.0;
				for (short k=0; k < input_dim; k++) {
					d += pow(ls[i * input_dim + k] / n[i] - ls[j * input_dim + k] / n[j], 2);
				}
				if (d < d_between_centroids_min) {
					d_between_centroids_min = d;
					d_between_centroids_argmin0 = i;
					d_between_centroids_argmin1 = j;
				}
			}
		}

		for (short i=0; i < n_centroids; i++) {
			if (n[i] < 2) {
				centroid_radii[i] = d_between_centroids_min;
			} else {
				d = 0.0;
				for (short k=0; k < input_dim; k++) {
					d += pow(maximum(0.0, ss[i * input_dim + k] / n[i] - pow(ls[i * input_dim + k] / n[i], 2)), 0.5);
				}
				centroid_radii[i] = microcluster_radius_alpha * d / input_dim;
				if (centroid_radii[i] == 0.0) {
					centroid_radii[i] = d_between_centroids_min;
				}
			}
		}

		d_min = pow(10, 8);
		int d_argmin = 0;
		for (short i=0; i < n_centroids; i++) {
			d = 0.0;
			for (short j=0; j < input_dim; j++) {
				d += pow(x[i * input_dim + j] - ls[i * input_dim + j] / n[i], 2);
			}
			if (d < d_min) {
				d_min = d;
				d_argmin = i;
			}
		}

		if (d_min <= centroid_radii[d_argmin]) {
			n[d_argmin] += 1;
			for (short i=0; i < input_dim; i++) {
				ls[d_argmin * input_dim + i] += x[i];
				ss[d_argmin * input_dim + i] += pow(x[i], 2);
			}
		} else {
			if (n_centroids < n_microclusters) {
				n[n_centroids] = 1;
				for (short i=0; i < input_dim; i++) {
					ls[n_centroids * input_dim + i] = x[i];
					ss[n_centroids * input_dim + i] = pow(x[i], 2);
				}
				n_centroids++;
			} else {
				n[d_between_centroids_argmin0] += n[d_between_centroids_argmin1];
				for (short i=0; i < input_dim; i++) {
					ls[d_between_centroids_argmin0 * input_dim + i] += ls[d_between_centroids_argmin1 * input_dim + i];
					ss[d_between_centroids_argmin0 * input_dim + i] += ss[d_between_centroids_argmin1 * input_dim + i];
				}
				n[d_between_centroids_argmin1] = 1;
				for (short i=0; i < input_dim; i++) {
					ls[d_between_centroids_argmin1 * input_dim + i] = x[i];
					ss[d_between_centroids_argmin1 * input_dim + i] = pow(x[i], 2);
				}
			}
		}

	}

	/*
	for(short i = 0; i < n_centroids; i++){
		std::cout << "centroid " << i << " = ";
		std::cout << n[i] << std::endl;
		for(short j = 0; j < input_dim; j++){
			std::cout << ls[i * input_dim + j] / n[i] << ',';
		}
		std::cout << std::endl;
	}
	*/

	if (n_train < n_max) {
		n_train++;
	}

}

void Clstrm::predict(float *x) {
	if (!is_score_thr_frozen) {
		_freeze_score_thr();
	}
	_cluster_point(x);
}

void Clstrm::validate(float *x) {

	if (!are_centroids_found) {
		_find_centroids();
	}

	if (is_score_thr_frozen) {
		_unfreeze_score_thr();
	}

	_cluster_point(x);

	if (n_val < n_max) {
		n_val++;
	}
}

float Clstrm::get_score_thr() {
	return score_thr[centroid_idx];
}

int Clstrm::get_score_n(int i) {
    return score_n[i];
}

float Clstrm::get_score_sum(int i) {
    return score_sum[i];
}

float Clstrm::get_score_ssum(int i) {
    return score_ssum[i];
}

int Clstrm::get_n_clusters() {
    return n_clusters;
}

void Clstrm::_freeze_score_thr() {
	for(short j = 0; j < n_clusters; j++){
		if (score_n[j] > 0) {
			score_thr[j] = score_alpha1 * score_sum[j] / score_n[j] + score_alpha2 * sqrt((maximum(0.0, score_ssum[j] - score_n[j] * pow(score_sum[j] / score_n[j], 2))) / score_n[j]);
		} else {
			score_thr[j] = 0.0;
		}
	}

	is_score_thr_frozen = true;
}

void Clstrm::_unfreeze_score_thr() {
	for (short i=0; i < n_clusters; i++) {
		score_n[i] = 0;
		score_sum[i] = 0.0;
		score_ssum[i] = 0.0;
		score_max[i] = 0.0;
		score_thr[i] = 0.0;
	}
    is_score_thr_frozen = false;
}

void Clstrm::_find_centroids() {

	int a;
	for (int i = 0; i < n_centroids; i++) {
		for (int j = i + 1; j < n_centroids; j++){
			if (n[microcluster_indexes_sorted[i]] < n[microcluster_indexes_sorted[j]]) {
				a = microcluster_indexes_sorted[i];
				microcluster_indexes_sorted[i] = microcluster_indexes_sorted[j];
				microcluster_indexes_sorted[j] = a;
			}
		}
	}

	if (n_centroids < n_clusters) {
		n_clusters = n_centroids;
	}

	for(short i = 0; i < n_clusters; i++){
		for(short j = 0; j < input_dim; j++){
			centroids[i * input_dim + j] = ls[microcluster_indexes_sorted[i] * input_dim + j] / n[microcluster_indexes_sorted[i]];
		}
	}

	bool have_centroids_changed;
	float d, d_min;
	int d_argmin;

	for(short iter = 0; iter < n_kmeans_iters; iter++){

		for(short i = 0; i < n_clusters; i++){
			centroid_n[i] = 0.0;
			for(short j = 0; j < input_dim; j++){
				centroids_new[i * input_dim + j] = 0.0;
			}
		}

		for(short i = 0; i < n_centroids; i++){
			d_min = pow(10, 8);
			for(short j = 0; j < n_clusters; j++){
				d = 0.0;
				for(short k = 0; k < input_dim; k++){
					d += pow(ls[i * input_dim + k] / n[i] - centroids[j * input_dim + k], 2);
				}
				if (d < d_min){
					d_min = d;
					d_argmin = j;
				}
			}
			for(short j = 0; j < input_dim; j++){
				centroids_new[d_argmin * input_dim + j] += ls[i * input_dim + j];
			}
			centroid_n[d_argmin] += n[i];
		}

		for(short i = 0; i < n_clusters; i++){
			for(short k = 0; k < input_dim; k++){
				centroids_new[i * input_dim + k] /= centroid_n[i];
			}
		}

		have_centroids_changed = false;

		for(short i = 0; i < n_clusters; i++){
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
			for(short j = 0; j < input_dim; j++){
				centroids[i * input_dim + j] = centroids_new[i * input_dim + j];
			}
		}

	}

	for(short i = 0; i < n_clusters; i++){
		for(short j = 0; j < input_dim; j++){
			centroids[i * input_dim + j] = centroids_new[i * input_dim + j];
		}
	}

	/*
	for(short i = 0; i < n_clusters; i++){
		std::cout << "centroid " << i << "(" << centroid_n[i] << ") = ";
		for(short j = 0; j < input_dim; j++){
			std::cout << centroids[i * input_dim + j] << ',';
		}
		std::cout << std::endl;
	}
	*/

	are_centroids_found = true;

}

void Clstrm::_cluster_point(float *x) {
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
