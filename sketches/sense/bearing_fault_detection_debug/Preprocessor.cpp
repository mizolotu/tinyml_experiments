/*
 * Preprocessor.cpp
 *
 *    Created on: Feb 14, 2023
 *            Author: mizolotu
 */

//#include <iostream>

#include "Preprocessor.h"
#include "DynamicDimensionQueue.h"
#include "utils.h"
#include <math.h>
#include "arduinoFFT.h"

Preprocessor::Preprocessor() {
    x_q = DynamicDimensionQueue(x_q_size, x_dim);
    fft_q = DynamicDimensionQueue(fft_q_size, fft_features * x_dim);
    x_mean = new float[x_dim];
    x_std = new float[x_dim];
    x_std_frozen = new float[x_dim];
    for (unsigned short j=0; j < x_dim; j++) {
        x_mean[j] = 0.0;
        x_std[j] = 0.0;
        x_std_frozen[j] = 0.0;
    }
    feature_n = 0;
    feature_vector = new float[fft_features * x_dim];
    feature_mean = new float[fft_features * x_dim];
    feature_std = new float[fft_features * x_dim];
    feature_sum = new float[fft_features * x_dim];
    feature_ssum = new float[fft_features * x_dim];
    for (unsigned short i=0; i < fft_features * x_dim; i++) {
        feature_vector[i] = 0.0;
        feature_mean[i] = 0.0;
        feature_std[i] = 0.0;
        feature_sum[i] = 0.0;
        feature_ssum[i] = 0.0;
    }
    FFT = arduinoFFT();
}

Preprocessor::~Preprocessor() {}

bool Preprocessor::put(float *x) {

    is_active = false;
    is_ready = false;

    float x_d = 0.0;
    float x_d_s = 0.0;
    float x_d_ss = 0.0;

    float x1[x_dim], x2[x_dim];
    for (unsigned short i=0; i < x_dim; i++) {
      x1[i] = x[i];
      x2[i] = 0.0;
    }

    x_q.enqueue(x1);

    if (is_baseline_calculated) {

        double re[fft_step], im[fft_step];
        float freq[fft_features * x_dim];

        for (unsigned short i=0; i < fft_step; i++) {
            re[i] = 0.0;
            im[i] = 0.0;
        }

        for (unsigned short i=0; i < fft_features * x_dim; i++) {
            freq[i] = 0.0;
        }

        x_q.mean(x_mean);
        x_q.std(x_std);

        x_d = 0.0;
        for (unsigned short i=0; i < x_dim; i++) {
            x_d += pow(x_std_frozen[i] > 0 ? (x[i] - x_mean[i]) / x_std_frozen[i] : 0.0, 2);
          //x_d += pow(x_std[i] > 0 ? (x[i] - x_mean[i]) / x_std[i] : 0.0, 2);
        }
        x_d = maximum(0.0, pow(x_d, 0.5));
        if (x_d > x_d_thr) {
            is_active = true;
        }

        if (is_active) {
            window_count = fft_step * fft_features;
        }

        if (window_count > 0) {

          if (f_step >= fft_step) {

            for (unsigned short j = 0; j < x_dim; j++) {
              for (unsigned short i = 0; i < fft_step; i++) {
                x_q.get(x_q.size() - fft_step + i, x2);
                re[i] = (x_std_frozen[j] == 0) ? 0.0 : (x2[j] - x_mean[j]) / x_std_frozen[j];
                //re[i] = (x_std[j] == 0) ? 0.0 : (x2[j] - x_mean[j]) / x_std[j];
                im[i] = 0;
              }
              FFT.Windowing(re, fft_step, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
              FFT.Compute(re, im, fft_step, FFT_FORWARD);
              FFT.ComplexToMagnitude(re, im, fft_step);

              for (unsigned short i = 0; i < fft_features; i++) {
                freq[i * x_dim + j] = re[i];
              }
            }

            fft_q.enqueue(freq);

            /*
              std::cout << "freq = ";
              for (unsigned short i = 0; i < fft_features * x_dim; i++) {
                std::cout << freq[i] << ", ";
              }
              std::cout << std::endl;
             */

              if (is_feature_std_found) {
                fft_q.xmax(feature_vector);

                /*
                std::cout << "feature vector = ";
                for (unsigned short i = 0; i < fft_features * x_dim; i++) {
                  std::cout << feature_vector[i] << ", ";
                }
                std::cout << std::endl;
                */

                for (unsigned short i = 0; i < fft_features * x_dim; i++) {
                  if (feature_std[i] == 0) {
                    feature_vector[i] = 0;
                  } else {
                    feature_vector[i] = (feature_vector[i] - feature_mean[i]) / feature_std[i];
                  }
                }

                /*
                std::cout << "feature vector std = ";
                for (unsigned short i = 0; i < fft_features * x_dim; i++) {
                  std::cout << feature_vector[i] << ", ";
                }
                std::cout << std::endl;
                */

                is_ready = true;

              } else {
                fft_q.xmax(feature_vector);
                feature_n++;
                for (unsigned short i = 0; i < fft_features * x_dim; i++) {
                  feature_sum[i] += feature_vector[i];
                  feature_ssum[i] += pow(feature_vector[i], 2);
                }
                f_step++;
                if (f_step == feature_steps) {
                  for (unsigned short i = 0; i < fft_features * x_dim; i++) {
                    feature_mean[i] = feature_sum[i] / feature_n;
                    feature_std[i] = sqrt(maximum(0.0, (feature_ssum[i] - feature_n * pow(feature_sum[i] / feature_n, 2)) / feature_n));
                  }

                  is_feature_std_found = true;

                  /*
                  for (unsigned short i = 0; i < fft_features * x_dim; i++) {
                    std::cout << feature_mean[i];
                    std::cout << ", ";
                  }
                  std::cout << "\n";
                  for (unsigned short i = 0; i < fft_features * x_dim; i++) {
                    std::cout << feature_std[i];
                    std::cout << ", ";
                  }
                  std::cout << "\n";
                  */
                }
              }
              window_count -= 1;

            } else {

              f_step += 1;

            }

        } else {

          // if window_count == 0, should we do anything?

        }

    } else {

        x_step++;

        if (x_step == baseline_steps) {

            x_q.std(x_std_frozen);
            x_q.mean(x_mean);

            for (unsigned short i=1; i < x_q.size(); i++) {
                x_q.get(i, x);
                    x_d = 0.0;
                    for (unsigned short j=0; j < x_dim; j++) {
                        x_d += pow(x_std_frozen[j] > 0 ? (x[j] - x_mean[j]) / x_std_frozen[j] : 0.0, 2);
                    }
                    x_d = maximum(0.0, pow(x_d, 0.5));
                    x_d_s += x_d;
                    x_d_ss += pow(x_d, 2);
            }
            x_d_ss = maximum(0.0, sqrt(maximum(0.0, (x_d_ss - (x_q.size() - 1) * pow(x_d_s / (x_q.size() - 1), 2)) / (x_q.size() - 1))));
            x_d_s = x_d_s / (x_q.size() - 1);
            x_d_thr = baseline_alpha1 * x_d_s + baseline_alpha2 * x_d_ss;

            is_baseline_calculated = true;

            //std::cout << "Distance from the sample to the mean threshold = " << x_d_thr << std::endl;

        }

    }

    return is_ready;

}

float* Preprocessor::get_feature_vector() {
    return feature_vector;
}

void Preprocessor::get_feature_vector(float* x) {
  for (unsigned short i = 0; i < fft_features * x_dim; i++) {
        x[i] = feature_vector[i];
    }
}

void Preprocessor::clear_qs() {
  f_step = 0;
  x_q.clear();
  fft_q.clear();
}
