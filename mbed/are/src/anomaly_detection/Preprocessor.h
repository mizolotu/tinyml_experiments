/*
 * Preprocessor.h
 *
 *  Created on: Feb 14, 2023
 *      Author: mizolotu
 */

#ifndef SRC_PREPROCESSOR_H_
#define SRC_PREPROCESSOR_H_

#include "DynamicDimensionQueue.h"
#include "arduinoFFT.h"

class Preprocessor {

	bool use_fft = false;

	short x_dim = 3;
	short x_q_size = 1024;
	short fft_n = 5;
	short fft_step = 8;
	short fft_features = 4;
	short fft_q_size = 32;
	short fv_q_size = 8;

	float baseline_alpha1 = 0.0;
	float baseline_alpha2 = 0.0;

	DynamicDimensionQueue x_q, fft_q, fv_q;
	arduinoFFT FFT;

	short x_step = 0;
	short f_step = 0;
	short baseline_steps = 1024;
	bool is_baseline_calculated = false;
	short feature_steps = 1024;
	bool is_feature_std_found = false;

	float *x_mean;
	float *x_std;
	float *x_std_frozen;
	float x_d_thr = 0.0;

	float *feature_vector, *feature_vector_;
	float *feature_mean;
	float *feature_std;
	float *feature_sum;
	float *feature_ssum;
	short feature_n;

	short fv_count = 0;

	bool is_active = false;
	bool is_ready = false;
	bool batch_mode = true;

	short window_count = 0;

	float eps = 1e-10;

	public:

		Preprocessor();
		Preprocessor(bool _use_fft);
		virtual ~Preprocessor();

		bool put(float *x);
		float* get_feature_vector();
		float* get_feature_vector(short i);
		void get_feature_vector(float *x);
		void clear_qs();
		short get_batch_size();
		bool get_batch_mode();
		void set_batch_mode(bool bm);

  private:

};

#endif /* SRC_PREPROCESSOR_H_ */
