#include "mbed.h"
#include "LSM9DS1.h"
#include "lorawan_emini.h"
#include "platform/CircularBuffer.h"

#include "model.h"

// Wind prediction

#define INPUT_SIZE 60  // needs to be same is MEASUREMENT_COUNT
#define DEBUG false

float current_sum;
float current[INPUT_SIZE];
float wind_speed[1];

//---------------//

#define TX_TIMER 60s
#define MEASUREMENT_TIMER 1s

Thread measure_thread;
Thread send_thread;
LW_Emini lwemini;

// Buffers for measurements
#define MEASUREMENT_COUNT 60
CircularBuffer<uint16_t, MEASUREMENT_COUNT> ain2_buf;
CircularBuffer<uint16_t, MEASUREMENT_COUNT> ain3_buf;

// Calibration values for measurements
float voltage_b = 0.85108956;
float current_k = 1.01964624;
float current_b = 0.26233607;

// ain0-ain3, all channels on same gpio clock
AnalogIn ain2(PB_3);
AnalogIn ain3(PB_4);

I2C i2c(I2C_SDA, I2C_SCL); // PB_7 PB_6 should also be good for i2c (i2c1). default pins i2c2
                           // SPI spi(SPI_MOSI, SPI_MISO, SPI_SCK); // mosi, miso, sclk

uint8_t lora_payload[6];  // Array for LoRa message payload

Mutex global_mutex;  // mutex to avoid race conditions between the threads

float calibrate_voltage(float voltage_recorded)
{
  float voltage_true = 0.0;
  if (voltage_recorded == 0.0) {
    voltage_true = 0.0;
  } else {
    voltage_true = voltage_recorded + voltage_b;
  }
  return voltage_true;
}

float calibrate_current(float current_recorded)
{
  float current_true = 0.0;
  if (current_recorded == 0.0) {
    current_true = 0.0;
  } else {
    current_true = current_k * current_recorded + current_b;
  }
  return current_true;
}

uint16_t get_average(CircularBuffer<uint16_t, MEASUREMENT_COUNT> buf)
{
    uint16_t count = buf.size();
    uint32_t sum = 0;
    uint16_t value;
    while(!buf.empty())
    {
        buf.pop(value);
        sum += value;
    }
    uint16_t avg = sum / count;
    return avg;
}

uint16_t convert2current_and_average(float* current, CircularBuffer<uint16_t, MEASUREMENT_COUNT> buf)
{
    uint16_t count = buf.size();
    uint32_t sum = 0;
    uint16_t value;
    int i = 0;
    while(!buf.empty())
    {
        buf.pop(value);
        current[i++] = calibrate_current(value * (3.3 / 65535.0) * (50 / 4.0));
        sum += value;
    }

    for(int n = i; n < MEASUREMENT_COUNT; n++) {
        current[i] = 0;
    }

    return sum / count; // Returns average of ADC values
}

void get_adc_measurements()
{
    while(1)
    {
        global_mutex.lock();

        uint16_t current_adc = ain2.read_u16();
        uint16_t voltage_adc = ain3.read_u16();

        // Debug prints for measurements

        float current = (current_adc * (3.3 / 65535)) / 0.08;
        float corrected_current = calibrate_current(current);
        float voltage = (voltage_adc * (3.3 / 65535))/ 0.025;
        float corrected_voltage = calibrate_voltage(voltage);

        debug("Current: %.4f\n", current);
        debug("Calibrated Current: %.4f\n", corrected_current);
        debug("Voltage: %.4f\n", voltage);
        debug("Calibrated Voltage: %.4f\n", corrected_voltage);

        ain2_buf.push(current_adc);
        ain3_buf.push(voltage_adc);

        global_mutex.unlock();
        
        ThisThread::sleep_for(MEASUREMENT_TIMER);
    }
}

void send_measurements()
{
  while (1)
  {
    global_mutex.lock();

    uint16_t analog_3 = get_average(ain3_buf);

    // Wind prediction
    uint16_t analog_2 = convert2current_and_average(current, ain2_buf);
    predict(current, wind_speed);
    uint16_t windspeed = uint16_t(wind_speed[0] * 1000);

    debug("analog channel 2: %d\n", analog_2);
    debug("analog channel 3: %d\n", analog_3);
    debug("predicted windspeed: %.2f\n", wind_speed[0]);
    debug("windspeed as x1000 int: %d\n", windspeed);
    
    // convert to int mV for lora message
    analog_2 = uint16_t(analog_2 * (3.3 / 65535) * 1000);
    analog_3 = uint16_t(analog_3 * (3.3 / 65535) * 1000);
    debug("Current measurement mV: %d\n", analog_2);
    debug("Voltage measurement mV: %d\n", analog_3);
    debug("Voltage: %.4f\n", (analog_3 / 1000.0 / 0.025));
    debug("Current: %.4f\n", (analog_2 / 1000.0 / 0.08));
    
    // Send measurements as ADC millivolts, need to be converted and calibrated
    // in the cloud.
    if (lwemini.joined)
    {
      lora_payload[0] = analog_2 >> 8 & 0xFF;
      lora_payload[1] = analog_2 & 0xFF;
      lora_payload[2] = analog_3 >> 8 & 0xFF;
      lora_payload[3] = analog_3 & 0xFF;
      lora_payload[4] = windspeed >> 8 & 0xFF;
      lora_payload[5] = windspeed & 0xFF;
      int16_t retcode = lwemini.send_message(lora_payload, sizeof(lora_payload));

      if (retcode == -1001)
      {
          debug("Message: ");
          for (uint8_t i : lora_payload)
          {
              debug("%x\n", i);
          }
          debug("\n");
          NVIC_SystemReset();
      }
    }
    
    global_mutex.unlock();

    ThisThread::sleep_for(TX_TIMER);
  }
}

int main(void)
{
  measure_thread.start(get_adc_measurements);
  send_thread.start(send_measurements);
  lwemini.init();
  while (1)
  {
  }

  return 0;
}
