#include <Wire.h>
#include <Adafruit_MCP4725.h>

#define RAMP_UP_TIME 5000  // in ms
#define ON_TIME 5000
#define RAMP_DOWN_TIME 5000
#define OFF_TIME 5000

#define TICK 100  // Time between voltage adjustments (ms)
#define UP_INC (4096 / (RAMP_UP_TIME / TICK))
#define DOWN_INC (4096 / (RAMP_DOWN_TIME / TICK))

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
  Serial.println("Ramping up...");
  while (speed < 4096) {
    MCP4725.setVoltage(speed, false);
    Serial.println(speed);
    speed += UP_INC;
    delay(TICK);
  }
  speed = 4096;
}

void rampDown() {
  Serial.println("Ramping down...");
  while (speed > 0) {
    MCP4725.setVoltage(speed, false);
    Serial.println(speed);
    speed -= DOWN_INC;
    delay(TICK);
  }
  speed = 0;
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  MCP4725.begin(0x60);    // Default I2C address of MCP4725 breakout board
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println("Running cycle:");
  rampUp();
  delay(ON_TIME);
  rampDown();
  delay(OFF_TIME);
  Serial.println("End of cycle");
}
