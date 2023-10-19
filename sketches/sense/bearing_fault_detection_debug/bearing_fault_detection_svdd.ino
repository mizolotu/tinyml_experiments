#include <math.h>
#include <Arduino_LSM9DS1.h>

#include "arduinoFFT.h"
#include "Preprocessor.h"
#include "Svdd.h"
#include "utils.h"

// main parameters

#define D_TRAIN                6     // training duration (minutes)
#define D_INF                  1     // inference duration (minutes)
#define D_SLEEP                1     // duration of the sleep between inference periods (minutes)
#define DELAY                 25     // delay during the inference and training (milliseconds)

// other parameters

#define VAL_SPLIT            0.3     // validation split (percentage of the training time)
#define DEBUG_MODE             1     // 0 - no output, 1 - print debug info

// constants

#define X_DIM                  3    // x, y, z

// counts

unsigned int total_count = 0;
unsigned int anomaly_count = 0;

// times

unsigned int t = 0;
unsigned int t_start = 0;
unsigned int t_stage_start = 0;

// durations in milliseconds

unsigned int d_val = D_TRAIN * VAL_SPLIT * 60 * 1000;
unsigned int d_train = D_TRAIN * 60 * 1000 - d_val;
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

void setup() {
  
  // start serial for debugging
  
  Serial.begin(115200);
  while (!Serial);

  // start IMU to get the accelerometer values

  if (!IMU.begin()) {
    while (1);
  }

  // start a timer

  if (DEBUG_MODE > 0) {
    Serial.println("Training...");
  }
  
  t_stage_start = millis();
  
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
            
      if (t - t_stage_start <= d_train) {
        
        model.train(prep.get_feature_vector());
        
      } else if (t - t_stage_start <= d_train + d_val) {
        
        model.validate(prep.get_feature_vector());
        
      } else {
        
        break;
        
      }
      
    }
    
    custom_delay(t_start);
       
  }

}

void loop() {
  
  _inference();
  delay(d_sleep);
  
}

void _inference() {

  if (DEBUG_MODE > 0) {
    Serial.println("Inferencing...");
  }

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

      if (score > score_thr) {

        // on lora, substitute the text below with the code to send the anomaly score ratio and/or other information
        //--------------------------------------------------------------------------------

        Serial.print("ANOMALY DETECTED (score to the threshold ratio: ");
        Serial.print(model.get_score() / model.get_score_thr(), 16); 
        Serial.print(")");
        Serial.println("");

        //--------------------------------------------------------------------------------
        
      }

    }

    custom_delay(t_start);

    // break when it is time

    t = millis();    
    if (t - t_stage_start > d_inf) {

      if (DEBUG_MODE > 0) {
        Serial.println("Sleeping...");
      }
      
      break;
    }
    
  }

  // clearing the queues

  prep.clear_qs();
  model.clear_score_q();

}

void custom_delay(unsigned int t_start) {
  
  t = millis();

  if (t - t_start < DELAY) {

    delay(DELAY - t + t_start);
  
  } else {
  
    if (DEBUG_MODE > 0) {
      Serial.println("");
      Serial.print("<Important: increase the DELAY value by at least ");
      Serial.print(t - t_start - DELAY);
      Serial.println(" milliseconds>");
      Serial.println("");
    }
    
  }
}
