#include <math.h>
#include <Arduino_LSM9DS1.h>

#include "arduinoFFT.h"
#include "DynamicDimensionQueue.h"
#include "Dense.h"
#include "Svdd.h"
#include "utils.h"


// main parameters


#define D_BASE              15     // duration of the baseline position evaluation (seconds)
#define D_TRAIN            120     // training duration (minutes)
#define D_INF                1     // inference duration (minutes)
#define D_SLEEP              3     // duration of the sleep between inference periods (minutes)
#define DELAY               25     // delay during the infrence and training (milliseconds)

// other parameters

#define P_STD                1     // percentage of the training time to calculate std coeffs
#define P_WARMUP             1     // percentage of the training time to calculate the center of the model (if 0, the center should be set manually)
#define P_VAL               30     // percentage of the training time to calculate the anomaly score threshold

#define XYZ_DIM              3     // xyz dimension for accelerometer data
#define XYZ_Q_SIZE        1024     // queue size for accelerometer data
#define FFT_N                5     // fft n
#define FFT_STEP             8     // fft step
#define FFT_FEATURES         4     // fft features
#define FFT_Q_SIZE          32     // fft queue size
#define BASE_STEPS        1000     // number of iteration to define the baseline
#define BASE_ALPHA         1.0     // baseline number of stds

#define N_LAYERS             3     // number of layers 
#define LAYER_1_UNITS       32     // the 1st layer units
#define LAYER_2_UNITS       32     // the 2nd layer units
#define LAYER_3_UNITS        2     // the 3rd layer units
#define LEARNING_RATE    0.003     // learning rate
#define MOMENTUM          0.01     // momentum
#define C_DEFAULT          0.0     // default c 

#define SCORE_ALPHA          1     // score threshold hyperparameter

#define DEBUG_MODE           3     // 0 - no output, 1 - print debug info, 2 - print data for the demo, 3 - print feature vectors

//stage

unsigned int stage = 0;

// trainng stages:
// 0 - estimating std coeffs
// 1 - calculating C for the model
// 2 - updating weights of the model
// 3 - calculating score threshold

// inference stages:
// 0 - inferencing
// 1 - sleeping

// counts

unsigned int fft_count = 0;
unsigned int sample_count = 0;
unsigned int total_count = 0;
unsigned int window_count = 0;
unsigned int anomaly_count = 0;

// times

unsigned int t = 0;
unsigned int t_start = 0;
unsigned int t_stage_start = 0;
unsigned int t_section_start = 0;

// durations in milliseconds

unsigned int d_base = D_BASE * 1000;
unsigned int d_std = D_TRAIN * P_STD * 600;
unsigned int d_warmup = D_TRAIN * P_WARMUP * 600;
unsigned int d_val = D_TRAIN * P_VAL * 600;
unsigned int d_train = D_TRAIN * 60 * 1000 - d_std - d_warmup - d_val;
unsigned int d_inf = D_INF * 60000;
unsigned int d_sleep = D_SLEEP * 60000;

// fft

arduinoFFT FFT = arduinoFFT();

// queues

DynamicDimensionQueue x_q(XYZ_Q_SIZE, XYZ_DIM);
DynamicDimensionQueue fft_q(FFT_Q_SIZE, FFT_FEATURES * XYZ_DIM);

// arrays

float x[XYZ_DIM];
float x_mean[XYZ_DIM];
float x_std[XYZ_DIM];
float x_std_frozen[XYZ_DIM];

float fft_mean[FFT_FEATURES * XYZ_DIM];
float fft_std[FFT_FEATURES * XYZ_DIM];

//double re[XYZ_Q_SIZE];
double re[FFT_STEP];
//double im[XYZ_Q_SIZE];
double im[FFT_STEP];
float freq[FFT_FEATURES * XYZ_DIM], freq_std[FFT_FEATURES * XYZ_DIM], feature_vector[FFT_FEATURES * XYZ_DIM];

// neural network

Dense layers[N_LAYERS] = {
  Dense(FFT_FEATURES * XYZ_DIM, LAYER_1_UNITS, &relu, &d_relu),
  Dense(LAYER_1_UNITS, LAYER_2_UNITS, &relu, &d_relu),
  
  //Dense(LAYER_2_UNITS, LAYER_3_UNITS, &tanh_, &d_tanh_)
  //Dense(LAYER_2_UNITS, LAYER_3_UNITS, &sigmoid, &d_sigmoid)
  
  Dense(LAYER_2_UNITS, LAYER_3_UNITS, &linear, &d_linear)
};

Svdd model = Svdd(N_LAYERS, layers, LEARNING_RATE, MOMENTUM, SCORE_ALPHA);

float c[LAYER_3_UNITS];

// other

float x_d;
float x_d_s;
float x_d_ss;
float x_d_thr;
bool is_active;
int idx;
int window_start = 0;
int seconds_remaining, seconds_remaining_last;

// model training

void setup() {
  
  // start serial for debugging
  
  Serial.begin(115200);
  while (!Serial);

  // start IMU to get the accelerometer values

  if (!IMU.begin()) {
    while (1);
  }

  Serial.println("Recoding the baseline... THE MOTOR SHOULD BE OFF!");
  
  t_stage_start = millis();
  
  while(1) {

    t_start = millis();

    if (DEBUG_MODE == 1) {
      seconds_remaining_last = _print_seconds(d_base, t, t_stage_start, seconds_remaining_last, sample_count);
    }  
  
    // get new xyz data point
  
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x[0], x[1], x[2]);
    }

    // add the new point to the baseline queue
       
    x_q.enqueue(x);
    
    // delay between iterations

    _delay(t_start);

    // end the stage

    if (t - t_stage_start > d_base) {
      break;
    }

    // increment counts

    sample_count += 1;
    total_count += 1;
    
  }

  // save xyz std

  x_q.std(x_std_frozen);
  
  // calculate action detection threshold
    
  x_q.mean(x_mean);
  
  for (unsigned short i=1; i < x_q.size(); i++) {
    x_q.get(i, x);
    x_d = 0.0;
    for (unsigned short j=0; j < XYZ_DIM; j++) {
      x_d += pow(x_std_frozen[j] > 0 ? (x[j] - x_mean[j]) / x_std_frozen[j] : 0.0, 2);
    }
    x_d = maximum(0.0, pow(x_d, 0.5));
    x_d_s += x_d;
    x_d_ss += pow(x_d, 2);
  }
  
  x_d_ss = maximum(0.0, sqrt(maximum(0.0, (x_d_ss - (x_q.size() - 1) * pow(x_d_s / (x_q.size() - 1), 2)) / (x_q.size() - 1))));
  x_d_s = x_d_s / (x_q.size() - 1);
  x_d_thr = x_d_s + BASE_ALPHA * x_d_ss;
  
  if (DEBUG_MODE > 0) {
    Serial.println("");
    Serial.println("THE MODEL TRAINING STARTS, THE MOTOR CAN NOW BE USED!");
    Serial.println("");
    
    if (DEBUG_MODE >= 2) {
      Serial.println("<Important: the baseline position has been found!>");
      Serial.print("<b=");
      Serial.print(x_d_thr, 6);
      Serial.print(">");
      Serial.println("");
    }
    
  } 
    
  // main training loop

  t_section_start = millis();
  t_stage_start = millis();

  sample_count = 0;

  while(1) {

    t_start = millis();

    // calculate mean

    x_q.mean(x_mean);
  
    // get new xyz data point
  
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x[0], x[1], x[2]);
    }

    // enque the point

    x_q.enqueue(x);

    if (DEBUG_MODE == 3) {
      Serial.print("<x=");
      for (unsigned short i = 0; i < XYZ_DIM; i++) {
        Serial.print(x[i], 16);
        (i < XYZ_DIM - 1) ? Serial.print(',') : Serial.print('>'); 
      }          
      Serial.println("");      
    }
 
    // check if there is activity

    is_active = false;
    /*for (unsigned short i=0; i<XYZ_DIM; i++) {
      if (abs(x[i] - x_mean[i]) > BASE_ALPHA * x_std_thr[i]) {
        is_active = true;
        break;
      }      
    }*/
    x_d = 0.0;
    for (unsigned short i=0; i < XYZ_DIM; i++) {
      x_d += pow(x_std_frozen[i] > 0 ? (x[i] - x_mean[i]) / x_std_frozen[i] : 0.0, 2);
    }
    x_d = maximum(0.0, pow(x_d, 0.5));
    if (x_d > x_d_thr) {
      is_active = true;
    }
    
    if (DEBUG_MODE == 2) {
      Serial.print("<d=");
      Serial.print(x_d, 6);
      Serial.print(">");
      Serial.println("");
    }
    
    // train only if there is an activity

    if (is_active) {
      window_count = FFT_STEP * FFT_FEATURES;  // i.e. at least during the next 32 loops the input will be analyzed
    }

    // check that there is enough samples to compute the first fft

    if (fft_count >= FFT_STEP) {

      // perform fft

      for (unsigned short j = 0; j < XYZ_DIM; j++) {

        for (unsigned short i = 0; i < FFT_STEP; i++) {
          
          x_q.get(x_q.size() - FFT_STEP + i, x);          
          re[i] = (x_std_frozen[j] == 0) ? 0.0 : (x[j] - x_mean[j]) / x_std_frozen[j];  // should we divide by std here or removing the baseline is enough?
          im[i] = 0;
          
        }
          
        FFT.Windowing(re, FFT_STEP, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
        FFT.Compute(re, im, FFT_STEP, FFT_FORWARD);
        FFT.ComplexToMagnitude(re, im, FFT_STEP);

        for (unsigned short i = 0; i < FFT_STEP / 2; i++) {
          freq[i * XYZ_DIM + j] = re[i];
        } 

      }

      // enqueue the resulting frequency vector to the fft queue

      fft_q.enqueue(freq);

      // check that it is still active

      if (window_count > 0) {

        if (stage == 0) {

          sample_count += 1;

          if (DEBUG_MODE == 1) {
            seconds_remaining_last = _print_seconds(d_std, t, t_stage_start, seconds_remaining_last, sample_count);
          }

        } else {

          // calculate and analyze features

          if (fft_q.isFull()) {
              
            fft_q.xmax(feature_vector);

            // standardize feature vector

            for (unsigned short i = 0; i < FFT_FEATURES * XYZ_DIM; i++) {
              if (fft_std[i] == 0) {
                feature_vector[i] = 0;
              } else {
                feature_vector[i] = (freq[i] - fft_mean[i]) / fft_std[i];
              }
            }

            // print the model inputs

            if (DEBUG_MODE == 3) {
              Serial.print("<i=");
              for (unsigned short i = 0; i < FFT_FEATURES * XYZ_DIM; i++) {
                Serial.print(feature_vector[i], 16);
                (i < FFT_FEATURES * XYZ_DIM - 1) ? Serial.print(',') : Serial.print('>'); 
              }          
              Serial.println("");
            }
            
            if (stage == 1) { 

              // update c

              for (int i=0; i<FFT_FEATURES * XYZ_DIM; i++) {
                model.set_inputs(feature_vector[i], i);
              }
              //model.forward(feature_vector);
              model.forward();

              if (DEBUG_MODE == 1) {                
                seconds_remaining_last = _print_seconds(d_warmup, t, t_stage_start, seconds_remaining_last, sample_count);
              }
          
            } else if (stage == 2) {

              // update weights and biases
          
              for (int i=0; i<FFT_FEATURES * XYZ_DIM; i++) {
                model.set_inputs(feature_vector[i], i);
              }
              //model.forward(feature_vector);
              model.forward();
              //model.backward(feature_vector);
              model.backward();

              if (DEBUG_MODE == 1) {
                
                seconds_remaining_last = _print_seconds(d_warmup, t, t_stage_start, seconds_remaining_last, sample_count);
                             
                Serial.print(" Loss = ");
                Serial.print(model.get_score(), 8);
                Serial.print(", outputs = ");
                for (unsigned short i = 0; i < model.get_output_dim(); i++) {
                  Serial.print(model.get_outputs(i), 8);
                  Serial.print(", ");
                }
                Serial.println("");
                
              }

            } else if (stage == 3) {

              // update only the score threshold

              for (int i=0; i<FFT_FEATURES * XYZ_DIM; i++) {
                model.set_inputs(feature_vector[i], i);
              }
              //model.forward(feature_vector);
              model.forward();

              if (DEBUG_MODE == 1) {              
                seconds_remaining_last = _print_seconds(d_val, t, t_stage_start, seconds_remaining_last, sample_count);              
              }
            
            } 

            // print the model's outputs and the score value

            if (DEBUG_MODE == 2) {
              
              Serial.print("<o=");
              for (unsigned short i = 0; i < LAYER_3_UNITS; i++) {
                Serial.print(model.get_outputs(i), 6);
                (i < LAYER_3_UNITS - 1) ? Serial.print(',') : Serial.print(';'); 
              }
              if (stage > 1 && sample_count > 0) {
                Serial.print("l=");
                Serial.print(model.get_score(), 6);
                Serial.print(">");
              }
              Serial.println("");
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

    t = millis();

    if (stage == 0 && t - t_stage_start > d_std) {

      fft_q.mean(fft_mean);
      fft_q.std(fft_std);

      if (DEBUG_MODE == 1) {
        Serial.println("");
        Serial.println("Standardization coefficients have been found! FFT mean:");
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
      } else if (DEBUG_MODE == 2) {
        Serial.println("<Important: standardization coefficients have been found!>");
      }
      Serial.println("");      

      t_stage_start = millis();
      sample_count = 0;
      stage = 1;
    
    } else if (stage == 1 && t - t_stage_start > d_warmup) {

      if (d_warmup > 0) {
        
        model.freeze_c();
      
      } else {
      
        for (unsigned short j = 0; j < LAYER_3_UNITS; j++) {
          c[j] = C_DEFAULT;
        }        
        model.freeze_c(c);
        
      }
    
      if (DEBUG_MODE == 1) {
        Serial.println("");  
        Serial.println("SVDD center has been found! The center value:");
        for (unsigned short j = 0; j < LAYER_3_UNITS; j++) {
          Serial.print(model.get_c(j));
          Serial.print(",");
        }
        Serial.println("");
      } else if (DEBUG_MODE == 2) {
          
        Serial.println("<Important: the model center has been found!>");
        Serial.println("<Important: training now...>");
          
        Serial.print("<c=");
        for (unsigned short j = 0; j < LAYER_3_UNITS; j++) {
          Serial.print(model.get_c(j));
          (j < LAYER_3_UNITS - 1) ? Serial.print(',') : Serial.print('>'); 
        }
      }
      Serial.println("");
      Serial.println("");
      
      t_stage_start = millis();
      sample_count = 0;
      stage = 2;
    
    } else if (stage == 2 && t - t_stage_start > d_train) {
  
      model.unfreeze_score_thr();

      if (DEBUG_MODE > 0) {
        Serial.println("");
        Serial.println("Training has been completed! Validating...");
        if (DEBUG_MODE == 2) {
          Serial.println("<Important: training has been completed!>");
          Serial.println("<Important: validating now...>");
        }        
        Serial.println("");
      }

      t_stage_start = millis();
      sample_count = 0;
      stage = 3;
    
    //} else if (stage == 3 && ((t - t_stage_start > d_val) || (t - t_section_start > d_std + d_warmup + d_train + d_val)) && sample_count >= FFT_Q_SIZE * N_MIN_VAL) {
    } else if (stage == 3 && t - t_stage_start > d_val) {

      model.freeze_score_thr();
      
      if (DEBUG_MODE == 1) {
        Serial.println("");
        Serial.println("Validation has been completed! Score threshold: ");
        Serial.print(model.get_score_thr(), 16);
        Serial.println("");
      } else if (DEBUG_MODE == 2) {
        Serial.print("<t=");
        Serial.print(model.get_score_thr(), 16);
        Serial.print(">");
        Serial.println("");
        Serial.println("<Important: validation has been completed!>");        
      }
      
      t_stage_start = millis();
      sample_count = 0;
      break;

    }
  
    total_count += 1;

    // delay between iterations

    _delay(t_start);
    
  }

}

void loop() {

  if (DEBUG_MODE == 1) {
     Serial.print("Inferencing for ");     
     Serial.print(D_INF);
     Serial.print(" minute(s)...");
     Serial.println("");
  } else if (DEBUG_MODE == 2) {
    Serial.println("<Important: inferencing now...>");
  }
    
  _inference();    
    
  if (DEBUG_MODE == 1) {
    Serial.println("");
    Serial.print("Sleeping for ");
    Serial.print(D_SLEEP);
    Serial.print(" minute(s)...");
    Serial.println("");
  } else if (DEBUG_MODE == 2) {
    Serial.println("<Important: sleeping now...>");
  }
  
  delay(d_sleep);  
  
}

void _inference() {

  t_stage_start = millis();

  fft_count = 0;
  window_count = 0;

  while(1) {

    t_start = millis();

    // calculate mean

    x_q.mean(x_mean);
  
    // get new xyz data point
  
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x[0], x[1], x[2]);
    }

    // enque the point

    x_q.enqueue(x);
 
    // check if there is activity

    is_active = false;

    /*for (unsigned short i=0; i<XYZ_DIM; i++) {
      if (abs(x[i] - x_mean[i]) > BASE_ALPHA * x_std_thr[i]) {
        is_active = true;
        break;
      }      
    }*/
    
    x_d = 0.0;
    for (unsigned short i=0; i < XYZ_DIM; i++) {
      x_d += pow(x_std_frozen[i] > 0 ? (x[i] - x_mean[i]) / x_std_frozen[i] : 0.0, 2);
    }
    x_d = maximum(0.0, pow(x_d, 0.5));
    if (x_d > x_d_thr) {
      is_active = true;
    }

    if (DEBUG_MODE == 2) {
      Serial.print("<d=");
      Serial.print(x_d, 6);
      Serial.print(">");
      Serial.println("");
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
          re[i] = (x_std_frozen[j] == 0) ? 0.0 : (x[j] - x_mean[j]) / x_std_frozen[j];  // should we divide by std here or removing the baseline is enough?
          im[i] = 0;
          
        }
          
        FFT.Windowing(re, FFT_STEP, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
        FFT.Compute(re, im, FFT_STEP, FFT_FORWARD);
        FFT.ComplexToMagnitude(re, im, FFT_STEP);

        // store the fft result in freq

        for (unsigned short i = 0; i < FFT_STEP / 2; i++) {
          freq[i * XYZ_DIM + j] = re[i];
        } 

      }
      
      // enqueue the resulting frequency vector to the fft queue

      fft_q.enqueue(freq);

      // check that it is still active

      if (window_count > 0) {

        // calculate and analyze features

        if (fft_q.isFull()) {
              
          fft_q.xmax(feature_vector);

          // standardize feature vector

          for (unsigned short i = 0; i < FFT_FEATURES * XYZ_DIM; i++) {
            if (fft_std[i] == 0) {
              feature_vector[i] = 0;
            } else {
              feature_vector[i] = (freq[i] - fft_mean[i]) / fft_std[i];
            }
          }

          // print feature vector standardized for debug purposes

          if (DEBUG_MODE == 2) {
            Serial.print("<i=");
            for (unsigned short i = 0; i < FFT_FEATURES * XYZ_DIM; i++) {
              Serial.print(feature_vector[i], 6);
              (i < FFT_FEATURES * XYZ_DIM - 1) ? Serial.print(',') : Serial.print('>'); 
            }          
            Serial.println("");
          }
          
          // inference
          
          for (int i=0; i<FFT_FEATURES * XYZ_DIM; i++) {
            model.set_inputs(feature_vector[i], i);
          }
          //model.forward(feature_vector);
          model.forward();

          if (DEBUG_MODE == 1 && model.get_score() > model.get_score_thr()) {
            Serial.print("ANOMALY DETECTED (score to the threshold ratio: ");
            Serial.print(model.get_score() / model.get_score_thr(), 16); 
            Serial.print(")");
          } else if (DEBUG_MODE == 2 && model.get_score() > model.get_score_thr()) {
            Serial.print("<Important: anomaly detected (score to the threshold ratio: ");
            Serial.print(model.get_score() / model.get_score_thr(), 16); 
            Serial.print(")>");          
            Serial.println("");
          }

          if (DEBUG_MODE == 2) {
            Serial.print("<o=");
            for (unsigned short i = 0; i < LAYER_3_UNITS; i++) {
              Serial.print(model.get_outputs(i), 6);
              (i < LAYER_3_UNITS - 1) ? Serial.print(',') : Serial.print(';'); 
            }          
            Serial.print("l=");
            Serial.print(model.get_score(), 6);
            Serial.print(">");
            Serial.println("");
          } else if (DEBUG_MODE == 3) {
            Serial.print("<i=");
            for (unsigned short i = 0; i < FFT_FEATURES * XYZ_DIM; i++) {
              Serial.print(feature_vector[i], 16);
              (i < FFT_FEATURES * XYZ_DIM - 1) ? Serial.print(',') : Serial.print('>'); 
            }          
            Serial.println("");
          }

          /*
           * In LORA, it makes sense to send the score to the threshold ratio, i.e. model.get_score() / model.get_score_thr()
           */
            
          sample_count += 1;

        }
      
        window_count -= 1;
        window_start += 1; 

      } else {
      
        window_start = 0;
      
      }
      
    } else {
    
      fft_count += 1;
    
    }

    total_count += 1;
 
    _delay(t_start);

    t = millis();
    if (t - t_stage_start > d_inf) {
      break;
    }
    
  }

  // clearing the queues

  x_q.clear();
  fft_q.clear();
  model.clear_score_q();

}

unsigned int _print_seconds(unsigned int duration, unsigned int t, unsigned int t_stage_start, unsigned int seconds_remaining_last, unsigned int sample_count) {
  if (duration + t_stage_start > t) {
    seconds_remaining = (d_base + t_stage_start - t) / 1000;
    if (seconds_remaining < 0) {
      seconds_remaining = 0;
    }
    if (seconds_remaining != seconds_remaining_last) {        
      Serial.print(seconds_remaining);
      Serial.print(" seconds remaining...");
      Serial.println("");
      seconds_remaining_last = seconds_remaining;
    }    
  }  
  return seconds_remaining_last;
}

void _delay(unsigned int t_start) {
  
  t = millis();    
  if (t - t_start < DELAY) {

    delay(DELAY - t + t_start);
  
  } else {
  
    if (DEBUG_MODE > 0 && DEBUG_MODE < 3) {
      Serial.println("");
      Serial.print("<Important: increase the DELAY value by at least ");
      Serial.print(t - t_start - DELAY);
      Serial.println(" milliseconds>");
      Serial.println("");
    }
    
  }
}
