#include <Wire.h>
#include <Adafruit_MCP4725.h>

#define RAMP_MIN_TIME    101
#define RAMP_MAX_TIME  15000
#define ON_TIME          100
#define OFF_TIME         100
#define TICK             100  // Time between voltage adjustments (ms)

Adafruit_MCP4725 MCP4725;

// Default pins:
// Arduino - MCP4725
// GND     - GND
// 5V      - VCC
// A4      - SDA
// A5      - SCL

// Outputs are the other GND and OUT pins of the MCP4725 DAC

int speed = 0;

void rampUp() {
  
  int ramp_time = RAMP_MIN_TIME + rand() % (RAMP_MAX_TIME - RAMP_MIN_TIME);  
  int speed_delta = (int) (4096 / (ramp_time / TICK));

  Serial.print("Ramping up for ");
  Serial.print(ramp_time);
  Serial.print(" milliseconds with step = ");
  Serial.print(speed_delta);
  Serial.print(":\n");
  
  while (speed < 4096) {
    MCP4725.setVoltage(speed, false);
    Serial.println(speed);
    speed += speed_delta;
    delay(TICK);
  }
  
  speed = 4096;
  
}

void rampDown() {
  
  int ramp_time = RAMP_MIN_TIME + rand() % (RAMP_MAX_TIME - RAMP_MIN_TIME);
  int speed_delta = (int) (4096 / (ramp_time / TICK));

  Serial.print("Ramping down for ");
  Serial.print(ramp_time);
  Serial.print(" milliseconds with step = ");
  Serial.print(speed_delta);
  Serial.print(":\n");  

  while (speed > 0) {
    MCP4725.setVoltage(speed, false);
    Serial.println(speed);
    speed -= speed_delta;
    delay(TICK);
  }
  
  speed = 0;
  
}

void setup() {
  
  Serial.begin(115200);
  MCP4725.begin(0x60);    // Default I2C address of MCP4725 breakout board
  
}

void loop() {
  
  rampUp();
  
  delay(ON_TIME);
  
  rampDown();  
  
  delay(OFF_TIME);
  
}
