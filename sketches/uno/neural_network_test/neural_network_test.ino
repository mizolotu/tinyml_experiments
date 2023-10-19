// ARDUINO UNO

#include <math.h>

#include "Network.h"
#include "Layer.h"
#include "Dense.h"
#include "Conv1d.h"
#include "utils.h"

// main parameters

#define D_TRAIN                1.0     // training interval duration (minutes)
#define D_INF                  1.0     // inference interval duration (minutes)

// other parameters

#define DEBUG_MODE               1     // 0 - no output, 1 - print debug info

// constants

#define X_DIM 60
#define Y_DIM 2

// times

unsigned long t = 0;
unsigned long t_stage_start = 0;
unsigned long n_train = 0;
unsigned long n_inf = 0;

// durations in milliseconds

float d_train = D_TRAIN * 60.0 * 1000.0;
float d_inf = D_INF * 60.0 * 1000.0;

// input array 

float x[X_DIM], y[Y_DIM];
float loss;

// model

/*
short n_channels = 1;
short n_filters = 1;
short kernel = 30;
short stride = 30;
short n_conv_output = ((X_DIM - kernel) / stride + 1) * n_filters;
short n_hidden2 = 1;

Layer* layers[] = {
  new Conv1d(X_DIM, n_channels, n_filters, kernel, stride, &relu, &d_relu),
  new Dense(n_conv_output, n_hidden2, &relu, &d_relu),
  new Dense(n_hidden2, Y_DIM, &linear, &d_linear)
};
*/

void setup() {
  
  // start serial for debugging
  
  Serial.begin(115200);
  while (!Serial);

  // model

  short n_hidden1 = 3;

  Layer* layers[] = {
    new Dense(X_DIM, n_hidden1, &relu, &d_relu),
    new Dense(n_hidden1, Y_DIM, &linear, &d_linear)
  };

  short n_layers = *(&layers + 1) - layers;

  Network model = Network(n_layers, layers);

  // training

  if (DEBUG_MODE > 0) {
    Serial.println("Training:");
  }

  _train(model);

  // inference  

  if (DEBUG_MODE > 0) {
    Serial.println("Inferencing:");
  }
  
  _inference(model);

  if (DEBUG_MODE > 0) {
    Serial.println("Done!");
  }

}

void loop() {  

}

void _train(Network model) {

  // start a timer
  
  t_stage_start = millis();
  
  while(1) {

    // get new xyz data point

    _generate_input_vector(x);
    _generate_output_vector(y);

    model.train(x, y);
    
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

void _inference(Network model) {

  // start a timer

  t_stage_start = millis();

  while(1) {
      
    // get new xyz data point

    _generate_input_vector(x);
    _generate_output_vector(y);
    
    model.evaluate(x, y);

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

void _generate_input_vector(float *r) {
  for (short i=0; i < X_DIM; i++) {
    r[i] = (rand() % 2000) / 1000.0 - 1;
  }
}

void _generate_output_vector(float *r) {
  for (short i=0; i < Y_DIM; i++) {
    r[i] = (rand() % 1000) / 1000.0;
  }
}
