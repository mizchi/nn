// BLAS (Accelerate.framework) bindings for MoonBit
// macOS only - uses Apple's vecLib/CBLAS

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// MoonBit FixedArray[Byte] is passed as pointer to data
typedef uint8_t *moonbit_bytes_t;

// ============================================================================
// Optimized: Allocate float buffers in C, operate in-place
// ============================================================================

// Create a float buffer of given size
float* blas_alloc_floats(int count) {
  return (float*)calloc(count, sizeof(float));
}

// Free a float buffer
void blas_free_floats(float* buf) {
  if (buf) free(buf);
}

// Copy floats from MoonBit FixedArray[Byte] to C buffer
void blas_copy_from_bytes(float* dst, moonbit_bytes_t src, int count) {
  memcpy(dst, src, count * sizeof(float));
}

// Copy floats from C buffer to MoonBit FixedArray[Byte]
void blas_copy_to_bytes(moonbit_bytes_t dst, float* src, int count) {
  memcpy(dst, src, count * sizeof(float));
}

// Get float at index from C buffer
float blas_get_float(float* buf, int idx) {
  return buf[idx];
}

// Set float at index in C buffer
void blas_set_float(float* buf, int idx, float val) {
  buf[idx] = val;
}

// ============================================================================
// Direct BLAS operations on C buffers (no copy overhead)
// ============================================================================

// sgemm on C buffers: C = alpha * A @ B + beta * C
void blas_sgemm_direct(
  int trans_a,
  int trans_b,
  int m, int n, int k,
  float alpha,
  float* a, int lda,
  float* b, int ldb,
  float beta,
  float* c, int ldc
) {
  cblas_sgemm(
    CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k,
    alpha,
    a, lda,
    b, ldb,
    beta,
    c, ldc
  );
}

// sgemv on C buffers: y = alpha * A @ x + beta * y
void blas_sgemv_direct(
  int trans,
  int m, int n,
  float alpha,
  float* a, int lda,
  float* x, int incx,
  float beta,
  float* y, int incy
) {
  cblas_sgemv(
    CblasRowMajor,
    trans ? CblasTrans : CblasNoTrans,
    m, n,
    alpha,
    a, lda,
    x, incx,
    beta,
    y, incy
  );
}

// saxpy on C buffers: y = alpha * x + y
void blas_saxpy_direct(int n, float alpha, float* x, int incx, float* y, int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}

// Add bias to each row of a matrix: out[b, j] += bias[j]
void blas_add_bias(float* out, float* bias, int batch, int dim) {
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < dim; j++) {
      out[b * dim + j] += bias[j];
    }
  }
}

// Apply ReLU in-place
void blas_relu_inplace(float* x, int n) {
  for (int i = 0; i < n; i++) {
    if (x[i] < 0.0f) x[i] = 0.0f;
  }
}

// Fused: matmul + bias + relu (for layer 1)
void blas_layer1_fused(
  float* input,   // batch x in_dim
  float* weight,  // in_dim x out_dim
  float* bias,    // out_dim
  float* output,  // batch x out_dim
  int batch, int in_dim, int out_dim
) {
  // output = input @ weight
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    batch, out_dim, in_dim,
    1.0f, input, in_dim, weight, out_dim, 0.0f, output, out_dim);
  // Add bias and ReLU
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < out_dim; j++) {
      float v = output[b * out_dim + j] + bias[j];
      output[b * out_dim + j] = v > 0.0f ? v : 0.0f;
    }
  }
}

// Fused: matmul + bias (for layer 2, no activation)
void blas_layer2_fused(
  float* input,   // batch x in_dim
  float* weight,  // in_dim x out_dim
  float* bias,    // out_dim
  float* output,  // batch x out_dim
  int batch, int in_dim, int out_dim
) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    batch, out_dim, in_dim,
    1.0f, input, in_dim, weight, out_dim, 0.0f, output, out_dim);
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < out_dim; j++) {
      output[b * out_dim + j] += bias[j];
    }
  }
}

// ============================================================================
// Batch Training Operations (backward pass + parameter update)
// ============================================================================

// Batch softmax: probs[b, k] = softmax(logits[b, :])
void blas_softmax_batch(float* logits, float* probs, int batch, int dim) {
  for (int b = 0; b < batch; b++) {
    float* logit_row = logits + b * dim;
    float* prob_row = probs + b * dim;
    // Find max for numerical stability
    float max_val = logit_row[0];
    for (int k = 1; k < dim; k++) {
      if (logit_row[k] > max_val) max_val = logit_row[k];
    }
    // Compute exp and sum
    float sum = 0.0f;
    for (int k = 0; k < dim; k++) {
      prob_row[k] = expf(logit_row[k] - max_val);
      sum += prob_row[k];
    }
    // Normalize
    for (int k = 0; k < dim; k++) {
      prob_row[k] /= sum;
    }
  }
}

// Compute loss and accuracy for a batch
// Returns: loss_sum in result[0], correct_count in result[1]
void blas_compute_loss_acc(
  float* probs,     // batch x output_dim
  int* labels,      // batch
  float* result,    // [loss_sum, correct_count]
  int batch, int output_dim
) {
  float loss_sum = 0.0f;
  int correct = 0;
  float eps = 1e-7f;
  for (int b = 0; b < batch; b++) {
    int label = labels[b];
    float* prob_row = probs + b * output_dim;
    loss_sum += -logf(prob_row[label] + eps);
    // Find argmax
    int pred = 0;
    float pred_val = prob_row[0];
    for (int k = 1; k < output_dim; k++) {
      if (prob_row[k] > pred_val) {
        pred = k;
        pred_val = prob_row[k];
      }
    }
    if (pred == label) correct++;
  }
  result[0] = loss_sum;
  result[1] = (float)correct;
}

// Complete training step for 2-layer MLP (forward + backward + update)
// This fused operation minimizes data movement
void blas_train_step(
  float* input,     // batch x input_dim
  int* labels,      // batch (labels for this batch)
  float* weight1,   // input_dim x hidden_dim
  float* bias1,     // hidden_dim
  float* weight2,   // hidden_dim x output_dim
  float* bias2,     // output_dim
  float* hidden,    // batch x hidden_dim (workspace)
  float* output,    // batch x output_dim (workspace)
  float* probs,     // batch x output_dim (workspace)
  float* grad_w1,   // input_dim x hidden_dim (workspace, zeroed on entry)
  float* grad_b1,   // hidden_dim (workspace, zeroed on entry)
  float* grad_w2,   // hidden_dim x output_dim (workspace, zeroed on entry)
  float* grad_b2,   // output_dim (workspace, zeroed on entry)
  float* delta2,    // batch x output_dim (workspace)
  float* delta1,    // batch x hidden_dim (workspace)
  float* result,    // [loss_sum, correct_count]
  float lr,
  int batch, int input_dim, int hidden_dim, int output_dim
) {
  // Forward pass: hidden = ReLU(input @ weight1 + bias1)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    batch, hidden_dim, input_dim,
    1.0f, input, input_dim, weight1, hidden_dim, 0.0f, hidden, hidden_dim);
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < hidden_dim; j++) {
      float v = hidden[b * hidden_dim + j] + bias1[j];
      hidden[b * hidden_dim + j] = v > 0.0f ? v : 0.0f;
    }
  }

  // Forward pass: output = hidden @ weight2 + bias2
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    batch, output_dim, hidden_dim,
    1.0f, hidden, hidden_dim, weight2, output_dim, 0.0f, output, output_dim);
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < output_dim; j++) {
      output[b * output_dim + j] += bias2[j];
    }
  }

  // Softmax and loss/accuracy
  blas_softmax_batch(output, probs, batch, output_dim);
  blas_compute_loss_acc(probs, labels, result, batch, output_dim);

  // Backward: delta2 = probs - one_hot(labels)
  for (int b = 0; b < batch; b++) {
    for (int k = 0; k < output_dim; k++) {
      delta2[b * output_dim + k] = probs[b * output_dim + k];
    }
    delta2[b * output_dim + labels[b]] -= 1.0f;
  }

  // grad_b2 = sum over batch of delta2
  for (int b = 0; b < batch; b++) {
    for (int k = 0; k < output_dim; k++) {
      grad_b2[k] += delta2[b * output_dim + k];
    }
  }

  // grad_w2 = hidden^T @ delta2
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    hidden_dim, output_dim, batch,
    1.0f, hidden, hidden_dim, delta2, output_dim, 0.0f, grad_w2, output_dim);

  // delta1 = delta2 @ weight2^T * relu_grad(hidden)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    batch, hidden_dim, output_dim,
    1.0f, delta2, output_dim, weight2, output_dim, 0.0f, delta1, hidden_dim);
  // Apply ReLU gradient (hidden > 0 ? 1 : 0)
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < hidden_dim; j++) {
      int idx = b * hidden_dim + j;
      if (hidden[idx] <= 0.0f) delta1[idx] = 0.0f;
    }
  }

  // grad_b1 = sum over batch of delta1
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < hidden_dim; j++) {
      grad_b1[j] += delta1[b * hidden_dim + j];
    }
  }

  // grad_w1 = input^T @ delta1
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    input_dim, hidden_dim, batch,
    1.0f, input, input_dim, delta1, hidden_dim, 0.0f, grad_w1, hidden_dim);

  // Update parameters: param -= lr/batch * grad
  float scale = lr / batch;
  cblas_saxpy(input_dim * hidden_dim, -scale, grad_w1, 1, weight1, 1);
  cblas_saxpy(hidden_dim, -scale, grad_b1, 1, bias1, 1);
  cblas_saxpy(hidden_dim * output_dim, -scale, grad_w2, 1, weight2, 1);
  cblas_saxpy(output_dim, -scale, grad_b2, 1, bias2, 1);
}

// Zero a float buffer
void blas_zero(float* buf, int n) {
  memset(buf, 0, n * sizeof(float));
}

// ============================================================================
// Int buffer operations (for labels)
// ============================================================================

// Create an int buffer of given size
int* blas_alloc_ints(int count) {
  return (int*)calloc(count, sizeof(int));
}

// Free an int buffer
void blas_free_ints(int* buf) {
  if (buf) free(buf);
}

// Set int at index in buffer
void blas_set_int(int* buf, int idx, int val) {
  buf[idx] = val;
}

// Get int at index from buffer
int blas_get_int(int* buf, int idx) {
  return buf[idx];
}

// ============================================================================
// Legacy API (with byte conversion) - kept for compatibility
// ============================================================================

void blas_sgemm(
  int trans_a, int trans_b,
  int m, int n, int k,
  float alpha,
  moonbit_bytes_t a_bytes, int lda,
  moonbit_bytes_t b_bytes, int ldb,
  float beta,
  moonbit_bytes_t c_bytes, int ldc
) {
  float *a = (float *)a_bytes;
  float *b = (float *)b_bytes;
  float *c = (float *)c_bytes;
  cblas_sgemm(CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void blas_sgemv(
  int trans, int m, int n,
  float alpha,
  moonbit_bytes_t a_bytes, int lda,
  moonbit_bytes_t x_bytes, int incx,
  float beta,
  moonbit_bytes_t y_bytes, int incy
) {
  cblas_sgemv(CblasRowMajor,
    trans ? CblasTrans : CblasNoTrans,
    m, n, alpha, (float*)a_bytes, lda, (float*)x_bytes, incx, beta, (float*)y_bytes, incy);
}

void blas_saxpy(int n, float alpha, moonbit_bytes_t x_bytes, int incx, moonbit_bytes_t y_bytes, int incy) {
  cblas_saxpy(n, alpha, (float*)x_bytes, incx, (float*)y_bytes, incy);
}

void blas_sscal(int n, float alpha, moonbit_bytes_t x_bytes, int incx) {
  cblas_sscal(n, alpha, (float*)x_bytes, incx);
}

float blas_sdot(int n, moonbit_bytes_t x_bytes, int incx, moonbit_bytes_t y_bytes, int incy) {
  return cblas_sdot(n, (float*)x_bytes, incx, (float*)y_bytes, incy);
}

float blas_snrm2(int n, moonbit_bytes_t x_bytes, int incx) {
  return cblas_snrm2(n, (float*)x_bytes, incx);
}

void blas_scopy(int n, moonbit_bytes_t x_bytes, int incx, moonbit_bytes_t y_bytes, int incy) {
  cblas_scopy(n, (float*)x_bytes, incx, (float*)y_bytes, incy);
}
