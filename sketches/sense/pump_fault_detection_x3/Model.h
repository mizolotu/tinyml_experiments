/*
 * Model.h
 *
 *  Created on: Aug 7, 2023
 *      Author: mizolotu
 */

#ifndef SRC_MODEL_H_
#define SRC_MODEL_H_

class Model {

	public:

		Model();
		virtual ~Model();

		virtual void train(float *x);
		virtual void validate(float *x);
		virtual void predict(float *x);
		virtual float get_score_thr();
		virtual int get_score_n();
		virtual int get_score_n(int i);
		virtual float get_score_sum();
		virtual float get_score_sum(int i);
		virtual float get_score_ssum();
		virtual float get_score_ssum(int i);
		virtual int get_n_clusters();

		float get_loss();
		float get_score();
		long get_n_train();
		long get_n_val();

	protected:

		float loss = 0.0;
		float score = 0.0;

		float score_alpha1 = 1.25;
		float score_alpha2 = 3.0;
		bool is_score_thr_frozen = true;

		long n_train = 0.0;
		long n_val = 0.0;
		long n_max = 2147483647;

		virtual void _freeze_score_thr();
		virtual void _unfreeze_score_thr();

};

#endif /* SRC_MODEL_H_ */
