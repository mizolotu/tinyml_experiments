#include "model.h"
#include "model_data.h"

#define INPUT_SIZE 60

#define DEBUG false


float current_sum;
float current[INPUT_SIZE] = {0.82589,0.52262,0.67711,0.70191,0.59891,0.62561,1.00137,0.91172,1.25123,0.91363,1.16159,1.16159,0.8793,1.112,1.37903,1.78912,1.40573,1.61554,1.20164,1.24551,1.24551,1.31609,2.19729,2.44716,2.14198,2.23544,1.74143,1.37521,1.37521,2.10955,2.11718,1.62508,1.5774,1.13679,1.38666,1.28175,1.28175,1.112,0.89646,0.81063,0.6485,0.48065,0.56076,0.67139,0.98992,0.7763,0.92889,0.69047,0.97276,0.75532,0.76676,0.76485,0.76485,0.57793,0.47493,0.45014,0.35286,0.55123,0.22125,0.31471};
float wind_speed[1];
bool new_data = false;

void setup() {

  if (!DEBUG) {
    for(unsigned short i= 0; i < INPUT_SIZE; i++) {
      current[i] = 0.0;
    }
  }  
  
  Serial.begin(115200);
  while (!Serial);
}

void loop() {    

  if (DEBUG) {

    // use debug current vector     

    delay(1000);
    new_data = true;
   
  } else {

    // read current from serial  

    static char buffer[32];
    static size_t pos;
    if (Serial.available()) {
      char c = Serial.read();
      if (c == '\n') {
        buffer[pos] = '\0';

        current_sum = 0.0;
        for(unsigned short i= 0; i < INPUT_SIZE - 1; i++) {
          current[i] = current[i + 1];
          current_sum += current[i];
        }
        
        current[INPUT_SIZE - 1] = atof(buffer);
        current_sum += current[INPUT_SIZE - 1];
        pos = 0;
        new_data = true;
      
      } else if (pos < sizeof buffer - 1) {  // otherwise, buffer it
        buffer[pos++] = c;
      }
    }

  }
  
  if (new_data) {

    if (current_sum > 0.0) {
      predict(current, wind_speed);
    } else {
      wind_speed[0] = y0_mean(0);
    }
    
    Serial.print("<current=");
    Serial.print(current[INPUT_SIZE - 1], 16);
    Serial.print(">");
    Serial.println("");

    Serial.print("<wind_speed=");
    Serial.print(*(wind_speed), 16);
    Serial.print(">");
    Serial.println("");
    
    new_data = false;
  }
  
}
