/*
 * Skmpp.h
 *
 *  Created on: Jun 2, 2023
 *      Author: mizolotu
 */

#ifndef SRC_SKMPP_H_
#define SRC_SKMPP_H_

class Skmpp {

	bool is_trainable = false;

	int feature_vector_count = 0;
	int batch_size = 4;
	int n_clusters = 3;
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
	float *centroid_radii;

	float loss = 0.0;

	int n_kmeans_iters = 10;

	public:
		Skmpp();
		virtual ~Skmpp();

		void _random_indexes_without_repetition(int imax, int n, int *indexes, float *cumsum_probs);

		void train(float *x);
		void validate();
		void predict(float *x);
		float get_loss();
};

#endif /* SRC_SKMPP_H_ */
