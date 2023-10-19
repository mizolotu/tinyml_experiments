#include <SPI.h>
#include <ADXL362.h>
#include <math.h>


#define DELAY                10
#define SERIES_LEN           32
#define INPUT_DIM             3
#define BATCH_SIZE           16
#define N_FEATURES            4
#define N_CLUSTERS            4
#define N_PROTOTYPES          4
#define INF_FREQ              4
#define TRAIN_ITERATIONS     30
#define VAL_ITERATIONS     1000
#define ALPHA                 3

 
ADXL362 xl;

int x, y, z, t;
int series[SERIES_LEN][INPUT_DIM];
int t_start, t_now;
short sample_count, feature_vector_count, count;
float batch[BATCH_SIZE][INPUT_DIM * N_FEATURES];
float xmin[INPUT_DIM * N_FEATURES];
float xmax[INPUT_DIM * N_FEATURES];
float centroids[N_CLUSTERS + N_PROTOTYPES][INPUT_DIM * N_FEATURES];
short n_centroids;
float batch_probs[BATCH_SIZE];
float centroid_probs[N_CLUSTERS + N_PROTOTYPES];
float cost, d; 
short prototype_dist_argmin;
short prototype_indexes[N_PROTOTYPES];
short centroid_indexes[N_CLUSTERS];
short stage;
short freq = SERIES_LEN;
float cluster_rads[N_CLUSTERS];
int cluster_n[N_CLUSTERS];
float cluster_s[N_CLUSTERS];
float cluster_ss[N_CLUSTERS];

void extract_features(int feature_series[SERIES_LEN][INPUT_DIM], float *feature_vector){
  
  float vmin[INPUT_DIM], vmax[INPUT_DIM], s[INPUT_DIM], ss[INPUT_DIM];
  
  for(short j = 0; j < INPUT_DIM; j++){
    vmin[j] = pow(10, 8);
    vmax[j] = -pow(10, 8);
    s[j] = 0;
    ss[j] = 0;
  }
  
  for(short i = 0; i < SERIES_LEN; i++){
    for(short j = 0; j < INPUT_DIM; j++){
      if (feature_series[i][j] < vmin[j]){
        vmin[j] = float(feature_series[i][j]);
      }
      if (series[i][j] > vmax[j]){
        vmax[j] = float(feature_series[i][j]);
      }
      s[j] += float(feature_series[i][j]);
      ss[j] += float(pow(feature_series[i][j], 2));
    }
  }
  
  for(short j = 0; j < INPUT_DIM; j++){
    feature_vector[j] = vmin[j];
    feature_vector[INPUT_DIM + j] = vmax[j];
    feature_vector[INPUT_DIM * 2 + j] = s[j] / (float)SERIES_LEN;
    feature_vector[INPUT_DIM * 3 + j] = (ss[j] - pow(s[j], 2) / (float)SERIES_LEN) / (float)SERIES_LEN;
    if (feature_vector[INPUT_DIM * 3 + j] < 0) {
      feature_vector[INPUT_DIM * 3 + j] = 0;
    } else {
      feature_vector[INPUT_DIM * 3 + j] = sqrt((ss[j] - pow(s[j], 2) / (float)SERIES_LEN) / (float)SERIES_LEN);
    }
  }
  
}


void random_indexes_without_repetition(short imax, short n, short *indexes, float *cumsum_probs){
  
  bool duplicate_found;
  float rnd;
  
  for(int i = 0; i < n; i++){
    
    duplicate_found = true;
    
    while(duplicate_found == true){
      
      rnd = random(1000)/1000.0;
      
      for (short j=0; j < imax; j++){
        if (rnd < cumsum_probs[j]){
          indexes[i] = j;
          break;
        }
      }
      
      duplicate_found = false;   
      
      for(int j = 0; j < i; j++){
        if(indexes[i] == indexes[j]){
          duplicate_found = true;
          break;
        }
      }
      
    }
    
  }
  
  for(short i = 0; i < n; i++){   
    for(short j = i+1; j < n; j++){
      if(indexes[i] > indexes[j]){
        rnd = indexes[i];
        indexes[i] = indexes[j];
        indexes[j] = rnd;
      }
    }
  }
    
}


void setup(){  
  
  Serial.begin(115200);
  while(!Serial);
  
  xl.begin(10);                   // Setup SPI protocol, issue device soft reset
  xl.beginMeasure();              // Switch ADXL362 to measure mode    
  
  for(short i = 0; i < SERIES_LEN; i++){
    xl.readXYZTData(x, y, z, t);
    series[i][0] = x;
    series[i][1] = y;
    series[i][2] = z;
    delay(DELAY);
  }
  
  feature_vector_count = 0;
  for(short j = 0; j < N_FEATURES * INPUT_DIM; j++){
    xmin[j] = pow(10, 8);
    xmax[j] = -pow(10, 8);
  }

  Serial.println("Stage: TRAINING");
    
}


void loop(){ 
  
  t_start = millis();   
    
  for(short i = 0; i < SERIES_LEN - 1; i++){
    for(short j = 0; j < INPUT_DIM; j++){
      series[i][j] = series[i+1][j];
    }
  }
  
  xl.readXYZTData(x, y, z, t);
  
  series[SERIES_LEN-1][0] = x;
  series[SERIES_LEN-1][1] = y;
  series[SERIES_LEN-1][2] = z;
    
  sample_count += 1;
  
  if (sample_count % freq == 0){
    
    extract_features(series, batch[feature_vector_count]);    

    if (stage == 0){ 

      // training

      feature_vector_count += 1;
    
      if (feature_vector_count % BATCH_SIZE == 0){

        for(short i = 0; i < BATCH_SIZE; i++){
          for(short j = 0; j < INPUT_DIM * N_FEATURES; j++){
            if (batch[i][j] < xmin[j]){
              xmin[j] = batch[i][j];
            }
            if (batch[i][j] > xmax[j]){
              xmax[j] = batch[i][j];
            }
          }
        }
      
        if (n_centroids == 0){

          for(short i = 0; i < BATCH_SIZE; i++){
            if (i == 0){
              batch_probs[i] = 1.0 / BATCH_SIZE;
            } else {
              batch_probs[i] = batch_probs[i-1] + 1.0 / BATCH_SIZE;
            }        
          }
        
          random_indexes_without_repetition(BATCH_SIZE, N_CLUSTERS, centroid_indexes, batch_probs);
        
          for(short i = 0; i < N_CLUSTERS; i++){
            for(short j = 0; j < INPUT_DIM * N_FEATURES; j++){
              centroids[i][j] = batch[centroid_indexes[i]][j];
            }          
          }
        
          n_centroids += N_CLUSTERS;
                
        } else {

          cost = 0.0;
          for(short i = 0; i < BATCH_SIZE; i++){
            batch_probs[i] = pow(10, 8);
            for(short j = 0; j < N_CLUSTERS; j++){
              d = 0.0; 
              for(short k = 0; k < INPUT_DIM * N_FEATURES; k++){
                d += pow((batch[i][k] - xmin[k]) / (xmax[k] - xmin[k] + pow(10, -8)) - (centroids[j][k] - xmin[k]) / (xmax[k] - xmin[k] + pow(10, -8)), 2);
              }
              if (d < batch_probs[i]){
                batch_probs[i] = d;
              }
            }
            cost += batch_probs[i];
          }
          cost = cost * (1.0 + 0.001 * BATCH_SIZE);        
          batch_probs[0] = (batch_probs[0] + 0.001 * cost) / cost;
          for(short i = 1; i < BATCH_SIZE; i++){
            batch_probs[i] = batch_probs[i-1] + (batch_probs[i] + 0.001 * cost) / cost;
          }
          random_indexes_without_repetition(BATCH_SIZE, N_PROTOTYPES, prototype_indexes, batch_probs);
  
          for(short i = 0; i < N_PROTOTYPES; i++){
            for(short j = 0; j < N_FEATURES * INPUT_DIM; j++){
              centroids[N_CLUSTERS + i][j] = batch[prototype_indexes[i]][j];
            }
          }

          for(short j = 0; j < N_CLUSTERS + N_PROTOTYPES; j++){
            centroid_probs[j] = 0;
          }
        
          for(short i = 0; i < BATCH_SIZE; i++){
            batch_probs[i] = pow(10, 8);
            for(short j = 0; j < N_CLUSTERS + N_PROTOTYPES; j++){
              d = 0.0; 
              for(short k = 0; k < INPUT_DIM * N_FEATURES; k++){
                d += pow((batch[i][k] - xmin[k]) / (xmax[k] - xmin[k] + pow(10, -8)) - (centroids[j][k] - xmin[k]) / (xmax[k] - xmin[k] + pow(10, -8)), 2);
              }
              if (d < batch_probs[i]){
                batch_probs[i] = d;
                prototype_dist_argmin = j;
              }
            }
            centroid_probs[prototype_dist_argmin]++;  
          }
          cost = BATCH_SIZE * (1 + 0.001 * (N_CLUSTERS + N_PROTOTYPES));
          centroid_probs[0] = (centroid_probs[0] + 0.001 * cost) / cost;
          for(short i = 1; i < N_CLUSTERS + N_PROTOTYPES; i++){
            centroid_probs[i] = centroid_probs[i-1] + (centroid_probs[i] + 0.001 * cost) / cost;
          }
          random_indexes_without_repetition(N_CLUSTERS + N_PROTOTYPES, N_CLUSTERS, centroid_indexes, centroid_probs);

          for(short i = 0; i < N_CLUSTERS; i++){
            for(short j = 0; j < N_FEATURES * INPUT_DIM; j++){
              centroids[i][j] = batch[centroid_indexes[i]][j];
            }
          }
        
          count++;
          
          if (count == TRAIN_ITERATIONS){

            count = 0;
            stage = 1;
            freq = INF_FREQ;
            
            Serial.println("Stage: VALIDATION");

            for(short i = 0; i < N_CLUSTERS; i++){
              cluster_s[i] = 0;
              cluster_ss[i] = 0;
            }

          }        
        }
      
        feature_vector_count = 0;
      
      }    
        
    } else if(stage == 1){

      // validation
      
      batch_probs[feature_vector_count] = pow(10, 8);
      for(short j = 0; j < N_CLUSTERS; j++){
        d = 0.0;
        for(short k = 0; k < INPUT_DIM * N_FEATURES; k++){
          d += pow((batch[feature_vector_count][k] - xmin[k]) / (xmax[k] - xmin[k] + pow(10, -8)) - (centroids[j][k] - xmin[k]) / (xmax[k] - xmin[k] + pow(10, -8)), 2);
        }
        if (d < batch_probs[feature_vector_count]){
          batch_probs[feature_vector_count] = d;
          prototype_dist_argmin = j;
        }
      }

      cluster_n[prototype_dist_argmin] += 1;
      cluster_s[prototype_dist_argmin] += batch_probs[feature_vector_count];
      cluster_ss[prototype_dist_argmin] += pow(batch_probs[feature_vector_count], 2);

      count++;      

      if (count == VAL_ITERATIONS){

        for(short i = 0; i < N_CLUSTERS; i++){
          cluster_ss[i] = (cluster_ss[i] - pow(cluster_s[i], 2) / (float)cluster_n[i]) / (float)cluster_n[i];
          if (cluster_ss[i] < 0){
            cluster_ss[i] = 0;
          }
          cluster_rads[i] = cluster_s[i] / (float)cluster_n[i] + ALPHA * sqrt(cluster_ss[i]);
        }
        
        count = 0;
        stage = 2;
        Serial.println("Stage: INFERENCE");

      }
       
    } else {

      // testing

      batch_probs[feature_vector_count] = pow(10, 8);
      for(short j = 0; j < N_CLUSTERS; j++){
        d = 0.0;
        for(short k = 0; k < INPUT_DIM * N_FEATURES; k++){
          d += pow((batch[feature_vector_count][k] - xmin[k]) / (xmax[k] - xmin[k] + pow(10, -8)) - (centroids[j][k] - xmin[k]) / (xmax[k] - xmin[k] + pow(10, -8)), 2);
        }
        if (d < batch_probs[feature_vector_count]){
          batch_probs[feature_vector_count] = d;
          prototype_dist_argmin = j;
        }
      }

      if (batch_probs[feature_vector_count] > cluster_rads[prototype_dist_argmin]){        
        Serial.println("Anomaly!");
        //Serial.println(batch_probs[feature_vector_count]);
        //Serial.println(cluster_rads[prototype_dist_argmin]);
        //Serial.print(batch_probs[feature_vector_count] / cluster_rads[prototype_dist_argmin] ? cluster_rads[prototype_dist_argmin] > 0 : 1);
      }
      
    }    

    sample_count = 0;
    
  }   
 
  t_now = millis();
  
  if (t_now - t_start < DELAY){
    delay(DELAY - t_now + t_start);
  }
  
}
