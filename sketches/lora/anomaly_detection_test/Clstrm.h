/*
 * Clstrm.h
 *
 *  Created on: Jun 6, 2023
 *      Author: mizolotu
 */

#ifndef SRC_CLSTRM_H_
#define SRC_CLSTRM_H_

class Clstrm {

	int n_cenroids = 0;

	int n_microclusters = 4;
	int n_clusters = 3;
	int input_dim = 12;
	int n_centroids = 0;

	float microcluster_radius_alpha = 3.0;

	float *n;
	float *ls;
	float *ss;

	float *centroids;
	float *centroids_new;
	float *centroid_radii;
	float *centroid_n;

	int *microcluster_indexes_sorted;

	float loss = 0.0;

	int n_kmeans_iters = 10;

	public:

		Clstrm();
		virtual ~Clstrm();

		void train(float *x);
		void validate();
		void predict(float *x);
		float get_loss();

};

#endif /* SRC_CLSTRM_H_ */
