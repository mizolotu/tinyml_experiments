#include "model.h"

#define SLEEP_INTERVAL 1000
#define PREDICT_AFTER     1

float x[INPUT_SIZE], y[1];
unsigned short step;

float x_test[INPUT_SIZE] = {1.17875000f,1.09864000f,0.74387000f,0.90600000f,0.50164000f,0.39482000f,0.58556000f,1.04715000f,1.17112000f,1.19783000f,1.00518000f,1.00137000f,1.03761000f,0.95178000f,0.61799000f,0.38147000f,0.54169000f,0.54169000f,0.43869000f,0.52452000f,0.42343000f,0.32234000f,0.16594000f,0.35477000f,0.20981000f,0.20981000f,0.13542000f,0.23079000f,0.30136000f,0.37384000f,0.58747000f,0.85259000f,0.76485000f,1.31990000f,1.31990000f,0.85069000f,1.12344000f,0.64850000f,1.06050000f,0.84878000f,1.33516000f,0.72861000f,0.70191000f,0.29755000f,0.49782000f,0.76867000f,0.69810000f,0.74196000f,0.68093000f,0.75913000f,0.61226000f,0.66567000f,0.59319000f,1.19211000f,0.95178000f,1.51255000f,1.40192000f,1.42862000f,0.94605000f,0.94605000f};


void setup() {

  Serial.begin(115200);
  while (!Serial);

  for (int i = 0; i < INPUT_SIZE; i++) {
    x[i] = 0.0;
  }

  step = 0;

}

void loop() {

  // move previous inputs one position to the left, append a new value to the end of the input array

  for (int i = 0; i < INPUT_SIZE - 1; i++) {
    x[i] = x[i + 1];
  }
  x[INPUT_SIZE - 1] = x_test[step]; // <--- here should go a new value (instead of x_test[step])

  // check whether it is time to predict or not

  if ((step + 1) % PREDICT_AFTER == 0) {

    predict(x, y);

    Serial.print(*(y), 8);
    Serial.println("");

  }

  step = (step + 1) % INPUT_SIZE;

  delay(SLEEP_INTERVAL);

}
