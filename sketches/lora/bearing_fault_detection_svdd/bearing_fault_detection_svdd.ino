#include <math.h>
#include <Arduino_LSM9DS1.h>

#include "arduinoFFT.h"
#include "Preprocessor.h"
#include "Svdd.h"
#include "utils.h"

// main parameters

#define D_TRAIN_TOTAL         24     // total training duration (hours)
#define D_TRAIN                1     // training interval duration (minutes)
#define D_INF                  1     // inference interval duration (minutes)
#define D_SLEEP                5     // sleep interval duration (minutes)
#define DELAY                 30     // delay during the inference and training (milliseconds)

// other parameters

#define VAL_SPLIT            0.3     // validation split (percentage of the training time)
#define DEBUG_MODE             1     // 0 - no output, 1 - print debug info

// constants

#define X_DIM                  3     // x, y, z

// times

unsigned int t = 0;
unsigned int t_start = 0;
unsigned int t_stage_start = 0;
unsigned int t_stage_interval_start = 0;

// durations in milliseconds

unsigned int d_val_total = D_TRAIN_TOTAL * VAL_SPLIT * 3600 * 1000;
unsigned int d_train_total = D_TRAIN_TOTAL * 3600 * 1000 - d_val_total;
unsigned int d_train = D_TRAIN * 60 * 1000;
unsigned int d_inf = D_INF * 60 * 1000;
unsigned int d_sleep = D_SLEEP * 60 * 1000;

// input array 

float x[X_DIM];

// preprocessor

Preprocessor prep = Preprocessor();

// model

Svdd model = Svdd();

// status of the preprocessor

bool status = false;

// score and score threshold

float score = 0.0; 
float score_thr = 0.0;
float score_to_thr_ratio = 0.0;
short n_scores = 0;

void setup() {
  
  // start serial for debugging
  
  Serial.begin(115200);
  while (!Serial);

  // start IMU to get the accelerometer values

  if (!IMU.begin()) {
    while (1);
  }

  // training start time

  t_stage_start = millis();

  while(1) {

    if (DEBUG_MODE > 0) {
      Serial.println("Training / validating...");
    }

    _train_and_validate(t_stage_start);

    if (DEBUG_MODE > 0) {
      Serial.println("Sleeping...");
    }

    // break when it is time

    t = millis();    
    if (t - t_stage_start > d_train_total + d_val_total) {

      if (DEBUG_MODE > 0) {
        Serial.print("Trained for ");
        Serial.print(model.get_n_train());
        Serial.print(" iterations!");
        Serial.println("");
        Serial.print("Validated for ");
        Serial.print(model.get_n_val());
        Serial.print(" iterations!");
        Serial.println("");
      }
      
      break;
    
    }
  
    delay(d_sleep);

  }

}

void loop() {

  if (DEBUG_MODE > 0) {
    Serial.println("Inferencing...");
  }
  
  _inference();

  if (DEBUG_MODE > 0) {
    Serial.println("Sleeping...");
  }
  
  delay(d_sleep);
  
}

void _train_and_validate(unsigned int t_stage_start) {

  // start a timer
  
  t_stage_interval_start = millis();
  
  while(1) {

    // start another timer

    t_start = millis();

    // get new xyz data point
  
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x[0], x[1], x[2]);
    }

    // add the new point to the preprocessor
       
    status = prep.put(x); // status = 1 (true) if the baseline position and standardization coefficients are calculated and there are some vibrations, otherwise - 0
    
    if (status) {
      
      t = millis();
            
      if (t - t_stage_start <= d_train_total) {
        
        model.train(prep.get_feature_vector());
        
      } else if (t - t_stage_start <= d_train_total + d_val_total) {
        
        model.validate(prep.get_feature_vector());
        
      } else {
       
        break;
        
      }
      
    }
    
    custom_delay(t_start);

    // break when it is time

    t = millis();    
    if (t - t_stage_interval_start > d_train) {      
      break;
    }

  }

  // clear the queues

  prep.clear_qs();
  model.clear_score_q();
  
}

void _inference() {

  // nulify score related counts

  score_to_thr_ratio = 0.0;
  n_scores = 0;

  // start a timer

  t_stage_start = millis();

  while(1) {
  
    // start another timer
    
    t_start = millis();

    // get new xyz data point
  
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x[0], x[1], x[2]);
    }

    // add the new point to the preprocessor

    status = prep.put(x);
    if (status) {      
      model.predict(prep.get_feature_vector());
      score = model.get_score();
      score_thr = model.get_score_thr();

      if (score_thr > 0) {

        score_to_thr_ratio += score / score_thr;
        n_scores += 1;
        
      }

    }

    custom_delay(t_start);

    // break when it is time

    t = millis();    
    if (t - t_stage_start > d_inf) {      
      break;
    }
    
  }

  // calculate the average score to the threshold ratio before sending it

  score_to_thr_ratio = n_scores > 0 ? score_to_thr_ratio / n_scores : 0.0;

  
  // uncomment "send_lora_msg" lines when deploying on e5 mini
  //--------------------------------------------------------------------------------

  Serial.print("Average score to the threshold ratio after the last inference period: ");
  Serial.print(score_to_thr_ratio, 16);  
  Serial.println("");

  if (score_to_thr_ratio > 1.0) {
    Serial.println("Sending LoRa message, anomaly detected\n");
    //send_lora_msg(true, score_to_thr_ratio);
  } else {
    Serial.println("Sending LoRa message, no anomaly detected\n");
    //send_lora_msg(false, score_to_thr_ratio);
  }


  //--------------------------------------------------------------------------------

  // clear the queues

  prep.clear_qs();
  model.clear_score_q();

}

void custom_delay(unsigned int t_start) {
  
  t = millis();

  if (t - t_start < DELAY) {

    delay(DELAY - t + t_start);
  
  } else {
  
    if (DEBUG_MODE > 0) {
      Serial.print("If this message appears frequently, increase the DELAY value by at least ");
      Serial.print(t - t_start - DELAY);
      Serial.print(" milliseconds!");
      Serial.println("");
    }
    
  }
}
