#define pgm_read_float_near(addr) (*(const float *)(addr))


const int INPUT_SIZE = 60;

const float x_mean_data[]  = {
0.33100782f
};
const float x_std_data[]  = {
0.61901443f
};
const float y0_mean_data[]  = {
0.80128182f
};
inline float x_mean(int i) {
return pgm_read_float_near(x_mean_data + i);
}

inline float x_std(int i) {
return pgm_read_float_near(x_std_data + i);
}

inline float y0_mean(int i) {
return pgm_read_float_near(y0_mean_data + i);
}

float score(float * input) {
    float var = 1.23;
    return var;
}

void predict(float* x, float* y) {

  float x_sum = 0.0;

  for (int i = 0; i < INPUT_SIZE; i++) {
    x_sum += x[i];
  }

  if (x_sum == 0) {

    y[0] = y0_mean(0);

  } else {

    float x_nrm[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
      x_nrm[i] = (x[i] - x_mean(0)) / x_std(0);
    }
    y[0] = score(x_nrm);
  }
}
