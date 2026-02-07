#ifdef __APPLE__
  #define ACCELERATE_NEW_LAPACK
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

#include <stdint.h>
#include <time.h>
#include <math.h>

void tensor_sgemm(
  int trans_a, int trans_b,
  int m, int n, int k, float alpha,
  const float* a, int lda,
  const float* b, int ldb,
  float beta, float* c, int ldc
) {
  cblas_sgemm(CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tensor_sgemm_offset(
  int trans_a, int trans_b,
  int m, int n, int k, float alpha,
  const float* a, int a_offset, int lda,
  const float* b, int b_offset, int ldb,
  float beta, float* c, int c_offset, int ldc
) {
  cblas_sgemm(CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k, alpha, a + a_offset, lda, b + b_offset, ldb,
    beta, c + c_offset, ldc);
}

// GELU forward: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void tensor_gelu_forward(const float* input, float* output, int n) {
  const float sqrt_2_pi = 0.7978845608028654f;
  const float coef = 0.044715f;
  for (int i = 0; i < n; i++) {
    float x = input[i];
    float inner = sqrt_2_pi * (x + coef * x * x * x);
    float t = tanhf(inner);
    output[i] = 0.5f * x * (1.0f + t);
  }
}

// GELU backward: dx[i] = dy[i] * gelu'(x[i])
void tensor_gelu_backward(const float* dy, const float* x_pre, float* dx, int n) {
  const float sqrt_2_pi = 0.7978845608028654f;
  const float coef = 0.044715f;
  for (int i = 0; i < n; i++) {
    float x = x_pre[i];
    float inner = sqrt_2_pi * (x + coef * x * x * x);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float inner_deriv = sqrt_2_pi * (1.0f + 3.0f * coef * x * x);
    float gelu_deriv = 0.5f * (1.0f + t) + 0.5f * x * sech2 * inner_deriv;
    dx[i] = dy[i] * gelu_deriv;
  }
}

// Softmax inplace over rows: data[row_offset .. row_offset+cols] for each row
void tensor_softmax_inplace(float* data, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    float* row = data + i * cols;
    float max_val = row[0];
    for (int j = 1; j < cols; j++) {
      if (row[j] > max_val) max_val = row[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
      row[j] = expf(row[j] - max_val);
      sum += row[j];
    }
    for (int j = 0; j < cols; j++) {
      row[j] /= sum;
    }
  }
}

// Log-softmax: result[i] = (x[i] - max) - log(sum(exp(x - max)))
void tensor_log_softmax(const float* input, float* output, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    const float* row_in = input + i * cols;
    float* row_out = output + i * cols;
    float max_val = row_in[0];
    for (int j = 1; j < cols; j++) {
      if (row_in[j] > max_val) max_val = row_in[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
      sum += expf(row_in[j] - max_val);
    }
    float log_sum = logf(sum);
    for (int j = 0; j < cols; j++) {
      row_out[j] = (row_in[j] - max_val) - log_sum;
    }
  }
}

uint64_t timer_clock_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
