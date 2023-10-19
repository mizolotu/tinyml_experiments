// ARDUINO SENSE

#include <math.h>

#include "Skmpp.h"
#include "Clstrm.h"
#include "Svdd.h"

#include "utils.h"

// main parameters

#define D_TRAIN                1.0     // training interval duration (minutes)
#define D_INF                  1.0     // inference interval duration (minutes)

// other parameters

#define DEBUG_MODE               1     // 0 - no output, 1 - print debug info

//#define RED 22

// constants

#define X_DIM                   12     // x length

// times

unsigned long t = 0;
unsigned long t_stage_start = 0;
unsigned long n_train = 0;
unsigned long n_inf = 0;

// durations in milliseconds

float d_train = D_TRAIN * 60.0 * 1000.0;
float d_inf = D_INF * 60.0 * 1000.0;

// input array 

float x[X_DIM];
float loss;

// model

//Skmpp model = Skmpp();
//Clstrm model = Clstrm();
Svdd model = Svdd();

void setup() {
  
  // start serial for debugging
  
  Serial.begin(115200);
  //while (!Serial);

  //pinMode(RED, OUTPUT);

  //digitalWrite(RED, LOW);
  //delay(1000);
  //digitalWrite(RED, HIGH);

  // training

  if (DEBUG_MODE > 0) {
    Serial.println("Training:");
  }

  _train();

  //digitalWrite(RED, LOW);
  //delay(1000);
  //digitalWrite(RED, HIGH);

  // validation

  _validate();

  // inference  

  if (DEBUG_MODE > 0) {
    Serial.println("Inferencing:");
  }
  
  _inference();

  //digitalWrite(RED, LOW);
  //delay(1000);
  //digitalWrite(RED, HIGH);

  if (DEBUG_MODE > 0) {
    Serial.println("Done!");
  }

}

void loop() {  

}

void _train() {

  // start a timer
  
  t_stage_start = millis();
  
  while(1) {

    // get new xyz data point

    _generate_feature_vector(x);

    model.train(x);
    
    loss = model.get_loss();
    
    if (DEBUG_MODE > 0) {
      //Serial.println(loss, 16);        
    }    

    n_train++;

    Serial.print(millis());
    Serial.print(",");
    Serial.print(n_train);
    Serial.println("");

    if (millis() - t_stage_start > d_train) {
      break;
    }
    
  }

}

void _validate() {
  model.validate();
}

void _inference() {

  // start a timer

  t_stage_start = millis();

  while(1) {
      
    // get new xyz data point

    _generate_feature_vector(x);
    
    model.predict(x);

    n_inf++;

    Serial.print(millis());
    Serial.print(",");
    Serial.print(n_inf);
    Serial.println("");       

    // break when it is time

    if (millis() - t_stage_start > d_inf) {
      break;
    }
    
  }

}

void _generate_feature_vector(float *r) {
  for (short i=0; i < X_DIM; i++) {
    r[i] = (rand() % 2000) / 1000.0 - 1;
  }
}
