/*
 * Clstrm.h
 *
 *  Created on: Jun 6, 2023
 *      Author: mizolotu
 */

#ifndef SRC_CLSTRM_H_
#define SRC_CLSTRM_H_

#include "Model.h"

class Clstrm : public Model {

	int n_cenroids = 0;

	int n_microclusters = 16;
	int n_clusters = 8;
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
	int n_kmeans_iters = 20;

	bool are_centroids_found = false;

	int centroid_idx = 0;

	int *score_n;
	float *score_sum;
	float *score_ssum;
	float *score_max;
	float *score_thr;

	public:

		Clstrm();
		virtual ~Clstrm();

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
		void _find_centroids();

};

#endif /* SRC_CLSTRM_H_ */
