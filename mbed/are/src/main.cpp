#include "mbed.h"
#include "LSM9DS1.h"
#include "lorawan_emini.h"
#include "platform/CircularBuffer.h"

#include "arduinoFFT.h"
#include "Preprocessor.h"
#include "Model.h"
#include "Skmpp.h"
#include "Clstrm.h"
#include "Svdd.h"
#include "utils.h"

//#include "ADXL362.h"
//#include "ADXL345.h"
//#include "ADXL345_I2C.h"

// Bearing fault detection
//------------------------------//

// main parameters

#define D_TRAIN_TOTAL         24     // total training duration (hours)
#define D_TRAIN                1     // training interval duration (minutes)
#define D_INF                  1     // inference interval duration (minutes)
#define D_SLEEP                4     // sleep interval duration (minutes)
#define DELAY                 18     // time interval between two subsequent iterations (milliseconds)

// other parameters

#define VAL_SPLIT           0.35     // validation split (percentage of the training time)
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

// models

Model* models[] = {new Svdd(), new Skmpp(), new Clstrm()};
int n_models = *(&models + 1) - models;

// status of the preprocessor

bool status = false;

// score and score threshold

float score = 0.0;
float score_thr = 0.0;

float score_to_thr_ratio = 0.0;
float* score_to_thr_ratio_avg = new float[n_models];
float* score_to_thr_ratio_max = new float[n_models];
short* n_scores = new short[n_models];

// other

bool is_slow = false;
unsigned int n_slow = 0;
bool cannot_read_imu = false;
unsigned int n_cannot_read = 0;

//-----------------------------//

Thread fault_detection_thread(osPriorityNormal, 1000);
LW_Emini lwemini;

uint8_t test_payload[13];

LowPowerTimer timer;  // Timer to use in training and inference

bool read_imu(float* x) {

  bool cannot_read = false;

  I2C i2c(I2C_SDA, I2C_SCL); // PB_7 PB_6 should also be good for i2c (i2c1). default pins i2c2
  //SPI spi(SPI_MOSI, SPI_MISO, SPI_SCK); // mosi, miso, sclk
  
  sleep_manager_lock_deep_sleep();

  LSM9DS1 imu(i2c);

  if (((imu.begin() >> 8) & 0xFF) != 0x68) {
    debug("Failed to communicate with LSM9DS1.\n");
    cannot_read = true;
  }

  //ADXL362 adxl362(SPI_CS, SPI_MOSI, SPI_MISO, SPI_SCK);
  //ADXL345  adxl345(SPI_MOSI, SPI_MISO, SPI_SCK, SPI_CS);
  //ADXL345_I2C adxl345(I2C_SDA, I2C_SCL);

  //uint8_t status;
  //int readings[3] = {0, 0, 0};
  
  //status = adxl362.read_status();
  //debug("Status: %d\r\n", status);

  //adxl362.reset();
  //ThisThread::sleep_for(600);

  //status = adxl362.read_status();
  //debug("Status: %d\r\n", status);

  //adxl362.frequency(1000000);

  //adxl362.set_mode(ADXL362::MEASUREMENT);
  //status = adxl362.read_status();
  //debug("Status: %d\r\n", status);

  //adxl345.setPowerControl(0x00);

  //Full resolution, +/-16g, 4mg/LSB.
  //adxl345.setDataFormatControl(0x0B);
    
  //3.2kHz data rate.
  //adxl345.setDataRate(ADXL345_3200HZ);

  //Measurement mode.
  //adxl345.setPowerControl(MeasurementMode);

  //uint16_t xx, yy, zz; 
  //while(1) {
    //xx = adxl362.scanx();
    //yy = adxl362.scany();
    //zz = adxl362.scanz();
    //debug("x = %d y = %d z = %d\r\n", xx, yy, zz);
    
    //adxl345.getOutput(readings);
        
    //13-bit, sign extended values.
    //printf("%i, %i, %i\n", (int16_t)readings[0], (int16_t)readings[1], (int16_t)readings[2]);
    
    //wait_ms(100);
    //ThisThread::sleep_for(100);
    //break;
  //}

  imu.setAccelODR(LSM9DS1::A_ODR_952);  // Set accelerator output data rate to 952 Hz
  
  // get new xyz data point
  imu.readAccel();
  x[0] = imu.ax;
  x[1] = imu.ay;
  x[2] = imu.az;

  //debug("x = %.16f, y = %.16f, z = %.16f\n", x[0], x[1], x[2]);

  sleep_manager_unlock_deep_sleep();

  return cannot_read;

}

bool custom_delay(unsigned int t_start) {
  
  t = timer.read_ms();

  bool slow = false;

  if (t - t_start < DELAY) {

    ThisThread::sleep_for(DELAY - t + t_start);
  
  } else {

    slow = true;
  
    if (DEBUG_MODE > 0) {
      debug("If this message appears frequently, increase the DELAY value by at least ");
      debug("%d", t - t_start - DELAY);
      debug(" milliseconds!");
      debug("\n");
    }
    
  }
  
  return slow;

}

void send_lora_msg(bool alert, float anomaly_ratio)
{
    debug("Sending LoRa message\n");
    // MSG: <1B bool: alert> <2B anomaly_ratio> <2B ACC AX> <2B ACC AY> <2B ACC AZ>
    float a[3];
    cannot_read_imu = read_imu(a);
    uint16_t ax[3] = {(uint16_t)(a[0]*10000), (uint16_t)(a[1]*10000), (uint16_t)(a[2]*10000)};
    debug("Accelerometer: x = %d, y = %d, z = %d\n\r", a[0], a[1], a[2]);
    uint16_t anomaly_score = uint16_t(anomaly_ratio * 1000);

    if (lwemini.joined)
    {
      test_payload[0] = alert;                          // 1 if alert, 0 otherwise
      test_payload[1] = (anomaly_score >> 8) & 0xFF;    // as 1000x converted to int
      test_payload[2] = (anomaly_score & 0xFF);
      test_payload[3] = ax[0] >> 8 & 0xFF;              // Accelerometer values as 10000x converted to int
      test_payload[4] = ax[0] & 0xFF;
      test_payload[5] = ax[1] >> 8 & 0xFF;
      test_payload[6] = ax[1] & 0xFF;
      test_payload[7] = ax[2] >> 8 & 0xFF;
      test_payload[8] = ax[2] & 0xFF;
      int16_t retcode = lwemini.send_message(test_payload, sizeof(test_payload));

      if (retcode == -1001)
      {
          debug("Message: ");
          for (uint8_t i : test_payload)
          {
              debug("%x\n", i);
          }
          debug("\n");
          //NVIC_SystemReset();
      }
    }
}

void send_lora_msg(bool alert, float* anomaly_ratio_avg, float *anomaly_ratio_max)
{
    debug("Sending LoRa message\n");
        
    uint16_t ar_a[3] = {(uint16_t)(anomaly_ratio_avg[0]*1000), (uint16_t)(anomaly_ratio_avg[1]*1000), (uint16_t)(anomaly_ratio_avg[2]*1000)};
    uint16_t ar_m[3] = {(uint16_t)(anomaly_ratio_max[0]*1000), (uint16_t)(anomaly_ratio_max[1]*1000), (uint16_t)(anomaly_ratio_max[2]*1000)};

    if (lwemini.joined)
    {
      test_payload[0] = alert;
      
      test_payload[1] = ar_a[0] >> 8 & 0xFF;
      test_payload[2] = ar_a[0] & 0xFF;
      
      test_payload[3] = ar_a[1] >> 8 & 0xFF;
      test_payload[4] = ar_a[1] & 0xFF;
      
      test_payload[5] = ar_a[2] >> 8 & 0xFF;
      test_payload[6] = ar_a[2] & 0xFF;
      
      test_payload[7] = ar_m[0] >> 8 & 0xFF;
      test_payload[8] = ar_m[0] & 0xFF;
      
      test_payload[9] = ar_m[1] >> 8 & 0xFF;
      test_payload[10] = ar_m[1] & 0xFF;
      
      test_payload[11] = ar_m[2] >> 8 & 0xFF;
      test_payload[12] = ar_m[2] & 0xFF;
      
      //debug("Avg scores: m1 = %d, m2 = %d, m3 = %d\n\r", ar_a[0], ar_a[1], ar_a[2]);
      //debug("Max scores: m1 = %d, m2 = %d, m3 = %d\n\r", ar_m[0], ar_m[1], ar_m[2]);

      int16_t retcode = lwemini.send_message(test_payload, sizeof(test_payload));

      if (retcode == -1001)
      {
          debug("Message: ");
          for (uint8_t i : test_payload)
          {
              debug("%x\n", i);
          }
          debug("\n");
          //NVIC_SystemReset();
      }
    }
}

void send_lora_msg(int d) {
    
  test_payload[0] = d;
  for (short i=1; i<13; i++) {
    test_payload[i] = 0;
  }

  /*
  debug("Sending LoRa message:\n");

  debug("Message:\n");
  for (uint8_t i : test_payload) {
    debug("%x\n", i);
  }
  debug("\n");
  */
        
  if (lwemini.joined) {      
    
    int16_t retcode = lwemini.send_message(test_payload, sizeof(test_payload));

  }
}

void _train_and_validate(uint16_t t_stage_start)
{
  // start a timer
  
  t_stage_interval_start = timer.read_ms();

  // nullify the max iteration duration

  n_slow = 0;
  n_cannot_read = 0;

  while(1) {

    // start another timer

    t_start = timer.read_ms();

    // get new xyz data point
  
    cannot_read_imu = read_imu(x);
    if (cannot_read_imu && n_cannot_read < 255) {
      n_cannot_read += 1;
    }

    // add the new point to the preprocessor
       
    status = prep.put(x); // status = 1 (true) if the baseline position and standardization coefficients are calculated and there are some vibrations, otherwise - 0
    
    if (status) {
      
      t = timer.read_ms();
            
      if (t - t_stage_start <= d_train_total) {
        
        for (short i=0; i < prep.get_batch_size(); i++) {
          for (short j=0; j < n_models; j++) {
            models[j]->train(prep.get_feature_vector(i));
          }
        }
        
      } else if (t - t_stage_start <= d_train_total + d_val_total) {
        
        for (short i=0; i < prep.get_batch_size(); i++) {
          for (short j=0; j < n_models; j++) {
            models[j]->validate(prep.get_feature_vector(i));
          }
        }
        
      } else {
       
        break;
        
      }
      
    } else {
      
      is_slow = custom_delay(t_start);
      if (is_slow && n_slow < 255) {
          n_slow += 1;
      }
      //debug("%i\n", n_slow);
    
    }
    
    // break when it is time

    t = timer.read_ms();    
    if (t - t_stage_interval_start > d_train) {      
      break;
    }

  }

  // clear the queues

  prep.clear_qs();

  // print score thr info

  for (short k=0; k<n_models; k++) {
    if (models[k]->get_n_clusters() == 0) {
        debug("Model %d score n = ", k);
        debug("%d\n", models[k]->get_score_n());
        debug("Model %d score sum = ", k);
        debug("%.6f\n", models[k]->get_score_sum());
        debug("Model %d score ssum = ", k);
        debug("%.6f\n", models[k]->get_score_ssum());
    } else {
		int n_clusters = models[k]->get_n_clusters();
		debug("Model %d score n = ", k);
		for (short l=0; l<n_clusters; l++) {
		  debug("%d, ", models[k]->get_score_n(l));
		}
		debug("\n");
		debug("Model %d score sum = ", k);
		for (short l=0; l<n_clusters; l++) {
		  debug("%.6f, ", models[k]->get_score_sum(l));
		}
		debug("\n");
		debug("Model %d score ssum = ", k);
		for (short l=0; l<n_clusters; l++) {
		  debug("%.6f, ", models[k]->get_score_ssum(l));
		}
		debug("\n");        
	}

  }    

  // send lora message with the delay info

  //send_lora_msg(n_slow);
  send_lora_msg(n_cannot_read);
  
}

void _inference() {

  // nulify score related counts

  for (short i=0; i < n_models; i++) {
    score_to_thr_ratio_avg[i] = 0.0;
    score_to_thr_ratio_max[i] = 0.0;
    n_scores[i] = 0;
  }

  // start a timer

  t_stage_start = timer.read_ms();

  while(1) {
  
    // start another timer
    
    t_start = timer.read_ms();

    // get new xyz data point
  
    cannot_read_imu = read_imu(x);

    // add the new point to the preprocessor

    status = prep.put(x);
    if (status) {      
      
      for (short i=0; i < n_models; i++) {
        
        models[i]->predict(prep.get_feature_vector());
      
        score = models[i]->get_score();
        score_thr = models[i]->get_score_thr();

        if (score_thr > 0) {

          score_to_thr_ratio_avg[i] += score / score_thr;
          score_to_thr_ratio_max[i] = (score / score_thr) > score_to_thr_ratio_max[i] ? score / score_thr : score_to_thr_ratio_max[i];
          n_scores[i] += 1;
        
        }
      
      }

    }

    is_slow = custom_delay(t_start);

    // break when it is time

    t = timer.read_ms();    
    if (t - t_stage_start > d_inf) {      
      break;
    }
    
  }

  for (short i=0; i < n_models; i++) {
    debug("Score threshold of the %d-th model = ", i);
    debug("%.6f", models[i]->get_score_thr());
    debug("\n");
  }

  // calculate the average score to the threshold ratio before sending it

  score_to_thr_ratio = 0.0;
  for (short i=0; i < n_models; i++) {
    score_to_thr_ratio_avg[i] = n_scores[i] > 0 ? score_to_thr_ratio_avg[i] / n_scores[i] : 0.0;
    score_to_thr_ratio += score_to_thr_ratio_avg[i];
  }
  score_to_thr_ratio /= n_models;

  // uncomment "send_lora_msg" lines when deploying on e5 mini
  //--------------------------------------------------------------------------------

  debug("Average score to the threshold ratios after the last inference period:\n");
  for (short i=0; i < n_models; i++) {
    debug("Model ");
    debug("%d", i);
    debug(": ");
    debug("%.16f", score_to_thr_ratio_avg[i]);  
    debug("\n");
  }

  debug("Maximum score to the threshold ratios after the last inference period:\n");
  for (short i=0; i < n_models; i++) {
    debug("Model ");
    debug("%d", i);
    debug(": ");
    debug("%.16f", score_to_thr_ratio_max[i]);  
    debug("\n");
  }

  if (score_to_thr_ratio > 1.0) {
    debug("Sending LoRa message, anomaly detected\n");
    send_lora_msg(true, score_to_thr_ratio_avg, score_to_thr_ratio_max);
  } else {
    debug("Sending LoRa message, no anomaly detected\n");
    send_lora_msg(false, score_to_thr_ratio_avg, score_to_thr_ratio_max);
  }


  //--------------------------------------------------------------------------------

  // clear the queues

  prep.clear_qs();

}

void run_fault_detection_loop()
{
  while(1) {

    if (DEBUG_MODE > 0) {
      debug("Inferencing...\n");
    }
  
    _inference();

    if (DEBUG_MODE > 0) {
      debug("Sleeping...\n");
    }
  
    ThisThread::sleep_for(d_sleep);

  }
}

int main(void)
{
  //--------------------------//
  // Setting up fault detection
  //--------------------------//

  lwemini.init();

  timer.start();  // Start timer used in training and inference
  
  // training start time

  t_stage_start = timer.read_ms();

  // switch batch mode on

  prep.set_batch_mode(true);

  while(1) {

    if (DEBUG_MODE > 0) {
      debug("Training / validating...\n");
    }

    _train_and_validate(t_stage_start);

    if (DEBUG_MODE > 0) {
      debug("Sleeping...\n");
    }

    // break when it is time

    t = timer.read_ms();    
    if (t - t_stage_start > d_train_total + d_val_total) {
      
      if (DEBUG_MODE > 0) {
        for (short i=0; i < n_models; i++) {
          debug("Model ");
          debug("%d", i);
          debug(" has been trained for ");
          debug("%d", models[i]->get_n_train());
          debug(" and validated for ");
          debug("%d", models[i]->get_n_val());
          debug(" iterations!");
          debug("\n");
        }
      }
      
      break;
    
    }

    ThisThread::sleep_for(d_sleep);

  }
  
  // switch batch mode off

  prep.set_batch_mode(false);
  
  //------------------------//

  debug("starting inference\n");
  fault_detection_thread.start(run_fault_detection_loop);
  debug("initializing lora\n");
  
  //lwemini.init();
  
  //while (1)
  //{
  //}

  return 0;
}
