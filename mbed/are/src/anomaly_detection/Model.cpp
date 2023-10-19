/*
 * Model.cpp
 *
 *  Created on: Aug 7, 2023
 *      Author: mizolotu
 */

#include "Model.h"

#include <math.h>
#include "utils.h"

#include <iostream>

Model::Model() {}

Model::~Model() {}

void Model::train(float *x) {}
void Model::validate(float *x) {}
void Model::predict(float *x) {}

float Model::get_score_thr() {
	return 0.0;
}

int Model::get_score_n() {
	return 0.0;
}

int Model::get_score_n(int i) {
	return 0.0;
}

float Model::get_score_sum() {
	return 0.0;
}

float Model::get_score_sum(int i) {
	return 0.0;
}

float Model::get_score_ssum() {
	return 0.0;
}

float Model::get_score_ssum(int i) {
	return 0.0;
}

float Model::get_loss() {
    return loss;
}

float Model::get_score() {
    return score;
}

long Model::get_n_train() {
	return n_train;
}

long Model::get_n_val() {
	return n_val;
}

int Model::get_n_clusters() {
	return 0;
}

void Model::_freeze_score_thr() {}
void Model::_unfreeze_score_thr() {}
