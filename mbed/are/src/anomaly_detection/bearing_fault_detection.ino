#include <math.h>
#include <Arduino_LSM9DS1.h>

#include "fix_fft_32k.h"
#include "DynamicDimensionQueue.h"
#include "Dense.h"
#include "Svdd.h"
#include "utils.h"

#define XYZ_DIM              3     // xyz dimension for accelerometer data
#define XYZ_Q_SIZE        1024     // queue size for accelerometer data
#define XYZ_SCALE         8192     // xyz scale 32767 / 4g (not sure if this is correct though)
#define FFT_N                5     // fft n
#define FFT_STEP             8     // fft step
#define FFT_FEATURES         4     // fft features
#define FFT_Q_SIZE          32     // fft queue size
#define BASELINE_STEPS    1000     // number of iteration to define the baseline
#define BASELINE_ALPHA       3     // baseline number of stds
#define INFERENCE_DELAY     10     // delay during the infrence and training

#define N_LAYERS             3     // number of layers 
#define LAYER_1_UNITS       64     // the 1st layer units
#define LAYER_2_UNITS       32     // the 2nd layer units
#define LAYER_3_UNITS       16     // the 3rd layer units
#define LEARNING_RATE     0.01     // learning rate

#define N_STD              500     // number of samples to calculate std coeffs
#define N_WARMUP          2000     // number of samples to calculate C
#define N_TRAIN          10000     // number of training samples
#define N_THR_VAL         2000     // number of threshold validation samples
#define N_COUNT_VAL       2000     // number of anomaly count validation samples

#define SCORE_ALPHA          3     // score threshold hyperparameter

#define DEBUG             true     // debug?
#define RECORD_DATA      false     // record data?

unsigned int stage = 0;

// Stages:
// 0 - estimating std coeffs
// 1 - calculating C for the model
// 2 - updating weights of the model
// 3 - calculating score threshold
// 4 - calculating anomalies count in a row
// 5 - inferencing

unsigned int fft_count = 0;
unsigned int sample_count = 0;
unsigned int total_count = 0;
unsigned int window_count = 0;
unsigned int anomaly_count = 0;

DynamicDimensionQueue x_q(XYZ_Q_SIZE, XYZ_DIM);
DynamicDimensionQueue fft_q(FFT_Q_SIZE, FFT_FEATURES * XYZ_DIM);

float x[XYZ_DIM];
float x_mean[XYZ_DIM];
float x_std[XYZ_DIM];
float x_std_thr[XYZ_DIM];

float fft_mean[FFT_FEATURES * XYZ_DIM];
float fft_std[FFT_FEATURES * XYZ_DIM];

bool is_active;

short re[XYZ_Q_SIZE];
short im[XYZ_Q_SIZE];
float freq[FFT_FEATURES * XYZ_DIM], freq_std[FFT_FEATURES * XYZ_DIM], feature_vector[FFT_FEATURES * XYZ_DIM] ;

int idx;
int window_start = 0;

Dense layers[N_LAYERS] = {
  Dense(FFT_FEATURES * XYZ_DIM, LAYER_1_UNITS, &relu, &d_relu),
  Dense(LAYER_1_UNITS, LAYER_2_UNITS, &relu, &d_relu),
  Dense(LAYER_2_UNITS, LAYER_3_UNITS, &sigmoid, &d_sigmoid)
};

Svdd model = Svdd(N_LAYERS, layers, LEARNING_RATE, SCORE_ALPHA);

void setup() {
  
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    if (DEBUG) {
      Serial.println("Failed to initialize IMU!");
    }
    while (1);
  }

  if (DEBUG) {
    Serial.println("Recoding the baseline... THE MOTOR SHOULD BE OFF!");
  }
  
  for (unsigned short i=0; i<XYZ_Q_SIZE; i++) {
  
    // get new xyz data point
  
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x[0], x[1], x[2]);
    }

    // scale the point
    
    for (unsigned short j=0; j<XYZ_DIM; j++) {
      x[j] *= XYZ_SCALE;
    }

    // add the new point to the baseline queue
       
    x_q.enqueue(x);

    total_count += 1;

    delay(INFERENCE_DELAY);    
    
  }

  // save xyz std

  x_q.std(x_std_thr);

  if (DEBUG) {
    Serial.println("Training the model... ACTIVATE THE MOTOR, BUT DO NOT MOVE THE BOARD!");
  }

}


void loop() {

  // calculate mean

  x_q.mean(x_mean);
  
  // get new xyz data point
  
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x[0], x[1], x[2]);
  }

  // scale the point

  for (unsigned short i=0; i<XYZ_DIM; i++) {
    x[i] *= XYZ_SCALE;
  }

  // enque the point

  x_q.enqueue(x);
 
  // check if there is activity

  is_active = false;

  for (unsigned short i=0; i<XYZ_DIM; i++) {
    if (abs(x[i] - x_mean[i]) > BASELINE_ALPHA * x_std_thr[i]) {
      is_active = true;
      break;
    }      
  }

  if (is_active) {
    window_count = FFT_STEP * FFT_FEATURES;  // i.e. at least during the next 32 loops the input will be analyzed
  }

  // check that there is enough samples to compute the first fft

  if (fft_count >= FFT_STEP) {

    // perform fft

    for (unsigned short j = 0; j < XYZ_DIM; j++) {

      for (unsigned short i = 0; i < FFT_STEP; i++) {
        x_q.get(x_q.size() - FFT_STEP + i, x);          
        re[i] = (x[j] - x_mean[j]); // / x_std_thr[j];  // should we divide by std here or removing the baseline is enough?
        im[i] = 0;
      }
          
      fix_fft(re, im, FFT_N, 0);

      idx = min(FFT_FEATURES - 1, window_start);
          
      if (window_start > FFT_FEATURES - 1) {
        for (unsigned short i = 0; i < FFT_FEATURES - 1; i++) {
          freq[i * XYZ_DIM + j] = freq[(i + 1) * XYZ_DIM + j];
        }
      } 

      // store the fft result in freq
          
      freq[idx * XYZ_DIM + j] = sqrt(re[0] * re[0] + im[0] * im[0]) / 2; // only the first frequency is used
    
    }

    // enqueue the resulting frequency vector to the fft queue

    fft_q.enqueue(freq);

    // check that it is still active

    if (window_count > 0) {

      if (stage == 0) {

        sample_count += 1;

        if (DEBUG) {
          Serial.print("Stage 1/5 (calculating standardization coefficients): ");
          Serial.print(sample_count);
          Serial.print("/");
          Serial.print(N_STD);
          Serial.println("");
        }
        
      } else {

        // print fft vector standardized for debug purposes

        if (RECORD_DATA) {
          Serial.print("<");
          for (unsigned short i = 0; i < FFT_FEATURES * XYZ_DIM; i++) {
            Serial.print((fft_std[i] == 0) ? 0 : (freq[i] - fft_mean[i]) / fft_std[i], 16);
            (i < FFT_FEATURES * XYZ_DIM - 1) ? Serial.print(',') : Serial.print('>'); 
          }          
          Serial.println("");
        }

        // calculate and analyze features

        if (fft_q.isFull()) {
              
          fft_q.xmax(feature_vector);
          //fft_q.mean(feature_vector);

          // standardize feature vector

          for (unsigned short i = 0; i < FFT_FEATURES * XYZ_DIM; i++) {
            if (fft_std[i] == 0) {
              feature_vector[i] = 0;
            } else {
              feature_vector[i] = (freq[i] - fft_mean[i]) / fft_std[i];
            }
          }

          if (stage == 1) { 

            // update c
          
            model.forward(feature_vector);

            if (DEBUG) {
              Serial.print("Stage 2/5 (calculating the center of the model): ");
              Serial.print(sample_count);
              Serial.print("/");
              Serial.print(N_WARMUP);              
              Serial.println("");
            }
          
          } else if (stage == 2) {

            // update weights and biases
          
            model.forward(feature_vector);
            model.backward(feature_vector);

            if (DEBUG) {
              Serial.print("Stage 3/5 (training the model): ");
              Serial.print(sample_count);
              Serial.print("/");
              Serial.print(N_TRAIN);
              Serial.print(" loss = ");
              Serial.print(model.get_score(), 16);
              Serial.println("");
            }

          } else if (stage == 3) {

            // update only the score threshold

            model.forward(feature_vector);

            if (DEBUG) {
              Serial.print("Stage 4/5 (calculating the anomaly score threshold): ");
              Serial.print(sample_count);
              Serial.print("/");
              Serial.print(N_THR_VAL);
              Serial.println("");
            }
            
          } else if (stage == 4) {

            // calculate anomaly count (score threshold should be frozen by this point)

            model.forward(feature_vector);

            if (DEBUG) {
              Serial.print("Stage 5/5 Calculating the anomaly count threshold): ");
              Serial.print(sample_count);
              Serial.print("/");
              Serial.print(N_COUNT_VAL);
              Serial.println("");
            }
            
          } else {

            // infrence
          
            model.forward(feature_vector);

            if (DEBUG && (model.get_score() > model.get_score_thr()) && (model.get_anomaly_count() >= model.get_anomaly_count_thr())) {
              Serial.print("ANOMALY DETECTED (score = ");
              Serial.print(model.get_score()); 
              Serial.print(", count = ");
              Serial.print(model.get_anomaly_count());
              Serial.print(")!");
              Serial.println("");
            }

            /*
             * In LORA, it makes sense to track both the score value and the anomaly count, i.e. 
             * - model.get_score()
             * - model.get_score_thr()
             * - model.get_anomaly_count()
             * - model.get_anomaly_count_thr()
             * 
             * However, currently the main criteria for a sample to be anomalous is anomaly count, 
             * i.e. if the count exceeds its threshold found during the validation stage, the sample 
             * is classified as anomalous. Therefore, the minimum amount of payload to be sent is simply: 
             * boolean (model.get_anomaly_count() > model.get_anomaly_count_thr()).
             */            
            
          }

          sample_count += 1;

        }
      
      }

      window_count -= 1;
      window_start += 1; 

    } else {
      
      window_start = 0;
      
    }
      
  } else {
    
    fft_count += 1;
    
  }

  if (stage == 0 && sample_count >= N_STD) {

    fft_q.mean(fft_mean);
    fft_q.std(fft_std);

    if (DEBUG) {
      Serial.println("");
      Serial.println("Standardization coefficients have been found!");
      Serial.println("FFT mean:");
      for (unsigned short j = 0; j < XYZ_DIM * FFT_FEATURES; j++) {
        Serial.print(fft_mean[j]);
        Serial.print(",");
      }
      Serial.println("");
      Serial.println("FFT std:");
      for (unsigned short j = 0; j < XYZ_DIM * FFT_FEATURES; j++) {
        Serial.print(fft_std[j]);
        Serial.print(",");
      }
      Serial.println("");
      Serial.println("");
    }

    sample_count = 0;
    stage = 1;
    
  } else if (stage == 1 && sample_count >= N_WARMUP) {

    model.freeze_c();    
    
    if (DEBUG) {  
      Serial.println("SVDD center has been found!");
      for (unsigned short j = 0; j < LAYER_2_UNITS; j++) {
        Serial.print(model.get_c(j));
        Serial.print(",");
      }
      Serial.println("");
      Serial.println("");
    }

    sample_count = 0;
    stage = 2;
    
  } else if (stage == 2 && sample_count >= N_TRAIN) {
  
    model.unfreeze_score_thr();

    if (DEBUG) {
      Serial.println("Training has been completed!");
      Serial.println("Validating...");
      Serial.println("");
      Serial.println("");
    }
    
    sample_count = 0;
    stage = 3;
    
  } else if (stage == 3 && sample_count >= N_THR_VAL) {

    model.freeze_score_thr();
    model.unfreeze_anomaly_count_thr();

    
    if (DEBUG) {
      Serial.println("Score threshold validation has been completed!");
      Serial.print("Score threshold: ");
      Serial.print(model.get_score_thr());
      Serial.println("");
      Serial.println("");
    }
    
    sample_count = 0;
    stage = 4;

  } else if (stage == 4 && sample_count >= N_COUNT_VAL) {

    model.freeze_anomaly_count_thr();
    
    if (DEBUG) {
      Serial.println("Anomaly count validation has been completed!");
      Serial.print("Anomaly count threshold: ");
      Serial.print(model.get_anomaly_count_thr());
      Serial.println("");
      Serial.println("");
    }
    
    sample_count = 0;
    stage = 5;

  }
  
  total_count += 1;

  if (DEBUG) {
    if ((total_count * INFERENCE_DELAY) % 60000 == 0) {
      Serial.println("");
      Serial.println("Time elapsed: ");
      Serial.print(total_count * INFERENCE_DELAY / 60000 );
      Serial.print(" minutes");
      Serial.println("");
      Serial.println("");
    }
  }
  
  delay(INFERENCE_DELAY);

}
