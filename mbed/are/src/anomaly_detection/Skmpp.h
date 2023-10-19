/*
 * Skmpp.h
 *
 *  Created on: Jun 2, 2023
 *      Author: mizolotu
 */

#ifndef SRC_SKMPP_H_
#define SRC_SKMPP_H_

#include "Model.h"

class Skmpp : public Model {

	bool is_trainable = false;

	int feature_vector_count = 0;
	int batch_size = 16;
	int n_clusters = 8;
	int input_dim = 12;
	int n_centroids = 0;

	float *batch;
	float *batch_probs;

	int *prototype_indexes;
	float *prototypes;
	float *prototype_weights;
	float *prototype_probs;
	int *prototype_indexes_sorted;

	int *centroid_indexes;
	float *centroids;
	float *centroids_new;
	float *centroid_weights;

	int centroid_idx = 0;
	int n_kmeans_iters = 10;

	int *score_n;
	float *score_sum;
	float *score_ssum;
	float *score_max;
	float *score_thr;

	public:

		Skmpp();
		virtual ~Skmpp();

		void train(float *x);
		void validate(float *x);
		void predict(float *x);
		float get_score_thr();
		int get_score_n(int i);
		float get_score_sum(int i);
		float get_score_ssum(int i);
		int get_n_clusters();

	private:

		void _freeze_score_thr();
		void _unfreeze_score_thr();

		void _cluster_point(float *x);
		void _random_indexes_without_repetition(int imax, int n, int *indexes, float *cumsum_probs);

};

#endif /* SRC_SKMPP_H_ */
