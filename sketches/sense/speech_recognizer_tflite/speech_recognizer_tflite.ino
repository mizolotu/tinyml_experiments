#include <fix_fft_32k.h>
#include <math.h>
#include <PDM.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

#define PDM_SOUND_GAIN    255   // sound gain of PDM mic
#define PDM_BUFFER_SIZE   256   // buffer size of PDM mic
#define FFT_SIZE          17    // number of buffers per fft
#define FFT_N             6     // fft n
#define FFT_FEATURES      33    // fft size 
#define SAMPLE_THRESHOLD  1000  // RMS threshold to trigger sampling
#define FEATURE_SIZE      7     // sampling size of one voice instance
#define TOTAL_COUNT_MAX   20000 // total number of voice instance
#define SCORE_THRESHOLD   0.5   // score threshold 

short sample[PDM_BUFFER_SIZE / 2];
short re[PDM_BUFFER_SIZE / 2 * FFT_SIZE];
short im[PDM_BUFFER_SIZE / 2 * FFT_SIZE];
float feature_vector[FEATURE_SIZE * FFT_FEATURES];
unsigned int sample_count = 0;
unsigned int total_count = 0;
float * prediction;
float no_probs[FEATURE_SIZE];
float yes_probs[FEATURE_SIZE];
float dunno_probs[FEATURE_SIZE];
float no_prob;
float yes_prob;
float dunno_prob;

tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));
const char* GESTURES[] = {
  "no",
  "yes",
  "silence"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

const float xmin_data[] PROGMEM = {
0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f
};
const float xmax_data[] PROGMEM = {
3950.00000000f,1414.00000000f,513.00000000f,217.00000000f,306.00000000f,187.00000000f,187.00000000f,145.00000000f,84.00000000f,95.00000000f,99.00000000f,94.00000000f,96.00000000f,97.00000000f,61.00000000f,88.00000000f,74.00000000f,54.00000000f,42.00000000f,64.00000000f,50.00000000f,33.00000000f,44.00000000f,41.00000000f,52.00000000f,46.00000000f,98.00000000f,50.00000000f,79.00000000f,89.00000000f,79.00000000f,63.00000000f,45.00000000f,3500.00000000f,3802.00000000f,3834.00000000f,1563.00000000f,1045.00000000f,451.00000000f,842.00000000f,672.00000000f,957.00000000f,753.00000000f,1128.00000000f,772.00000000f,525.00000000f,395.00000000f,554.00000000f,622.00000000f,337.00000000f,248.00000000f,156.00000000f,126.00000000f,73.00000000f,71.00000000f,69.00000000f,67.00000000f,255.00000000f,63.00000000f,91.00000000f,62.00000000f,62.00000000f,61.00000000f,60.00000000f,60.00000000f,60.00000000f,2905.00000000f,3471.00000000f,5032.00000000f,3744.00000000f,1609.00000000f,1480.00000000f,1988.00000000f,2041.00000000f,731.00000000f,657.00000000f,684.00000000f,2177.00000000f,1423.00000000f,876.00000000f,1718.00000000f,692.00000000f,437.00000000f,441.00000000f,417.00000000f,482.00000000f,367.00000000f,373.00000000f,159.00000000f,325.00000000f,485.00000000f,322.00000000f,231.00000000f,95.00000000f,95.00000000f,127.00000000f,93.00000000f,93.00000000f,100.00000000f,4665.00000000f,4561.00000000f,8820.00000000f,3183.00000000f,2179.00000000f,1476.00000000f,1775.00000000f,1417.00000000f,1782.00000000f,1041.00000000f,1139.00000000f,1030.00000000f,1228.00000000f,1040.00000000f,835.00000000f,531.00000000f,596.00000000f,723.00000000f,289.00000000f,415.00000000f,956.00000000f,538.00000000f,907.00000000f,581.00000000f,541.00000000f,1311.00000000f,664.00000000f,381.00000000f,383.00000000f,234.00000000f,258.00000000f,141.00000000f,115.00000000f,2911.00000000f,4251.00000000f,4329.00000000f,5122.00000000f,2452.00000000f,4688.00000000f,1574.00000000f,2965.00000000f,1968.00000000f,1093.00000000f,1014.00000000f,1631.00000000f,1294.00000000f,688.00000000f,784.00000000f,1015.00000000f,1105.00000000f,867.00000000f,621.00000000f,944.00000000f,726.00000000f,445.00000000f,790.00000000f,672.00000000f,654.00000000f,485.00000000f,682.00000000f,601.00000000f,1473.00000000f,1195.00000000f,1031.00000000f,1068.00000000f,247.00000000f,3019.00000000f,5796.00000000f,3274.00000000f,3982.00000000f,2917.00000000f,2479.00000000f,3103.00000000f,1846.00000000f,747.00000000f,671.00000000f,579.00000000f,704.00000000f,697.00000000f,602.00000000f,647.00000000f,674.00000000f,450.00000000f,611.00000000f,878.00000000f,987.00000000f,690.00000000f,565.00000000f,2362.00000000f,1395.00000000f,956.00000000f,1015.00000000f,695.00000000f,240.00000000f,771.00000000f,775.00000000f,412.00000000f,216.00000000f,276.00000000f,2817.00000000f,2030.00000000f,1927.00000000f,7449.00000000f,1754.00000000f,2219.00000000f,1212.00000000f,983.00000000f,946.00000000f,936.00000000f,1075.00000000f,827.00000000f,953.00000000f,866.00000000f,342.00000000f,657.00000000f,600.00000000f,813.00000000f,573.00000000f,668.00000000f,731.00000000f,403.00000000f,997.00000000f,512.00000000f,1397.00000000f,1891.00000000f,990.00000000f,620.00000000f,1040.00000000f,800.00000000f,838.00000000f,213.00000000f,270.00000000f
};

inline float xmin(int i) {
    return pgm_read_float_near(xmin_data + i);
}

inline float xmax(int i) {
    return pgm_read_float_near(xmax_data + i);
}

void onPDMdata() {
  short sample_buffer[PDM_BUFFER_SIZE];
  int bytes_available = PDM.available();
  PDM.read(sample_buffer, bytes_available);
  for (unsigned short i = 0; i < (bytes_available / 2); i++) {
    sample[i] = sample_buffer[i];
  }
}


void setup() {

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  
  tflInterpreter->AllocateTensors();

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.begin(115200);
  while (!Serial);

  PDM.onReceive(onPDMdata);
  PDM.setBufferSize(PDM_BUFFER_SIZE);
  PDM.setGain(PDM_SOUND_GAIN);

  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }  

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  for (unsigned short i = 0; i < FEATURE_SIZE; i++) {
    for (unsigned short j = 0; j < FFT_SIZE; j++) {
      delay(8);
      for (unsigned short k = 0; k < PDM_BUFFER_SIZE / 2; k++) {
        re[j * PDM_BUFFER_SIZE / 2 + k] = sample[k];
        im[j * PDM_BUFFER_SIZE / 2 + k] = 0;
      }
    }
    fix_fft(re, im, FFT_N, 0);
    for (unsigned short j = 0; j < FFT_FEATURES; j++) {
      feature_vector[i * FFT_FEATURES + j] = (int)(sqrt(re[j] * re[j] + im[j] * im[j]) / 2);
    }
  }
  
  digitalWrite(LED_BUILTIN, LOW);  
  
}


void loop() {

  // copy previous samples 
  
  for (unsigned short i = 0; i < FEATURE_SIZE - 1; i++) {
    for (unsigned short j = 0; j < FFT_FEATURES; j++) {
      feature_vector[i * FFT_FEATURES + j] = feature_vector[(i + 1) * FFT_FEATURES + j];
    }
    no_probs[i] = no_probs[i + 1];
    yes_probs[i] = yes_probs[i + 1];
    dunno_probs[i] = dunno_probs[i + 1];
  }  

  // calculate new samples

  for (unsigned short j = 0; j < FFT_SIZE; j++) {
    delay(8);
    for (unsigned short k = 0; k < PDM_BUFFER_SIZE / 2; k++) {
      re[j * PDM_BUFFER_SIZE / 2 + k] = sample[k];
      im[j * PDM_BUFFER_SIZE / 2 + k] = 0;
    }
  }
  fix_fft(re, im, FFT_N, 0);
  for (unsigned short j = 0; j < FFT_FEATURES; j++) {
    feature_vector[(FEATURE_SIZE - 1) * FFT_FEATURES + j] = (int)(sqrt(re[j] * re[j] + im[j] * im[j]) / 2);
  }

  // make prediction

  for (unsigned short i = 0; i < FFT_FEATURES * FEATURE_SIZE; i++) {
    tflInputTensor->data.f[i] = (feature_vector[i] - xmin(i)) / (xmax(i) - xmin(i));
  }   
  
  //prediction = predict(feature_vector);

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    while (1);
    return;
  }
  for (int i = 0; i < NUM_GESTURES; i++) {
    Serial.print(GESTURES[i]);
    Serial.print(": ");
    Serial.println(tflOutputTensor->data.f[i], 6);
  }
  Serial.println();

  no_prob = tflOutputTensor->data.f[0];
  yes_prob = tflOutputTensor->data.f[1];
  dunno_prob = tflOutputTensor->data.f[2];

  /*
  no_probs[FEATURE_SIZE - 1] = tflOutputTensor->data.f[0];
  yes_probs[FEATURE_SIZE - 1] = tflOutputTensor->data.f[1];
  dunno_probs[FEATURE_SIZE - 1] = tflOutputTensor->data.f[2];
  
  no_prob = 0;
  yes_prob = 0;
  dunno_prob = 0;  
  for (unsigned short i = 0; i < FEATURE_SIZE; i++) {
    no_prob += no_probs[i];
    yes_prob += yes_probs[i];
    dunno_prob += dunno_probs[i];
  }
  */

  /*
  Serial.print(no_prob);
  Serial.print(", ");
  Serial.print(yes_prob);
  Serial.print(", ");
  Serial.print(dunno_prob);
  Serial.println();
  */
  
  if (yes_prob > no_prob && yes_prob > dunno_prob) {
    
    Serial.print("yes:");
    Serial.print(" ");
    
    Serial.print(no_prob / FEATURE_SIZE);
    Serial.print(", ");
    Serial.print(yes_prob / FEATURE_SIZE);
    Serial.print(", ");
    Serial.print(dunno_prob / FEATURE_SIZE);
    Serial.println("");
    
    digitalWrite(LEDR,HIGH);
    digitalWrite(LEDG,LOW);
    digitalWrite(LEDB,HIGH);
    
  } else if (no_prob > yes_prob && no_prob > dunno_prob) {
    
    Serial.print("no:");    
    Serial.print(" ");
    
    Serial.print(no_prob);
    Serial.print(", ");
    Serial.print(yes_prob);
    Serial.print(", ");
    Serial.print(dunno_prob);
    Serial.println("");
    
    digitalWrite(LEDR,LOW);
    digitalWrite(LEDG,HIGH);
    digitalWrite(LEDB,HIGH);
    
  } else {
    
    Serial.print("dunno:");
    Serial.print(" ");       
    
    Serial.print(no_prob);
    Serial.print(", ");
    Serial.print(yes_prob);
    Serial.print(", ");
    Serial.print(dunno_prob);
    Serial.println("");
    
    digitalWrite(LEDR,HIGH);
    digitalWrite(LEDG,HIGH);
    digitalWrite(LEDB,LOW);

  }
  
  //delay(1000);

  total_count++;
  if (total_count >= TOTAL_COUNT_MAX) {
    PDM.end();
    digitalWrite(LEDR,HIGH);
    digitalWrite(LEDG,HIGH);
    digitalWrite(LEDB,HIGH);
    while (1);
  }
  
}
