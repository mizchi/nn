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

// LayerNorm forward: output = gamma * (x - mean) * rstd + beta
// Stores mean/rstd per row for backward pass
void tensor_layer_norm_fwd(
  const float* input, const float* gamma, const float* beta,
  float* output, float* mean_out, float* rstd_out,
  int outer, int last_dim, float eps
) {
  float inv_dim = 1.0f / (float)last_dim;
  for (int i = 0; i < outer; i++) {
    const float* row = input + i * last_dim;
    float* out_row = output + i * last_dim;
    float mean = 0.0f;
    for (int j = 0; j < last_dim; j++) mean += row[j];
    mean *= inv_dim;
    float var = 0.0f;
    for (int j = 0; j < last_dim; j++) {
      float d = row[j] - mean;
      var += d * d;
    }
    var *= inv_dim;
    float rstd = 1.0f / sqrtf(var + eps);
    mean_out[i] = mean;
    rstd_out[i] = rstd;
    for (int j = 0; j < last_dim; j++) {
      out_row[j] = gamma[j] * (row[j] - mean) * rstd + beta[j];
    }
  }
}

// LayerNorm backward: computes dx, accumulates d_gamma, d_beta
void tensor_layer_norm_bwd(
  const float* dy, const float* x, const float* mean, const float* rstd,
  const float* gamma, float* dx, float* d_gamma, float* d_beta,
  int outer, int last_dim
) {
  float inv_dim = 1.0f / (float)last_dim;
  for (int i = 0; i < outer; i++) {
    const float* dy_row = dy + i * last_dim;
    const float* x_row = x + i * last_dim;
    float* dx_row = dx + i * last_dim;
    float m = mean[i];
    float rs = rstd[i];
    // Accumulate d_gamma, d_beta and compute sums
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int j = 0; j < last_dim; j++) {
      float x_hat = (x_row[j] - m) * rs;
      d_gamma[j] += dy_row[j] * x_hat;
      d_beta[j] += dy_row[j];
      float dy_gamma = dy_row[j] * gamma[j];
      sum1 += dy_gamma;
      sum2 += dy_gamma * x_hat;
    }
    sum1 *= inv_dim;
    sum2 *= inv_dim;
    for (int j = 0; j < last_dim; j++) {
      float x_hat = (x_row[j] - m) * rs;
      float dy_gamma = dy_row[j] * gamma[j];
      dx_row[j] = rs * (dy_gamma - sum1 - x_hat * sum2);
    }
  }
}

// Reshape [batch, seq, num_heads*d_k] -> [batch, num_heads, seq, d_k]
void tensor_reshape_for_heads(
  const float* src, float* dst,
  int batch, int seq, int num_heads, int d_k
) {
  int d_model = num_heads * d_k;
  for (int b = 0; b < batch; b++) {
    for (int s = 0; s < seq; s++) {
      for (int h = 0; h < num_heads; h++) {
        const float* in_ptr = src + b * seq * d_model + s * d_model + h * d_k;
        float* out_ptr = dst + b * num_heads * seq * d_k + h * seq * d_k + s * d_k;
        for (int d = 0; d < d_k; d++) {
          out_ptr[d] = in_ptr[d];
        }
      }
    }
  }
}

// Reshape [batch, num_heads, seq, d_k] -> [batch, seq, num_heads*d_k]
void tensor_reshape_from_heads(
  const float* src, float* dst,
  int batch, int seq, int num_heads, int d_k
) {
  int d_model = num_heads * d_k;
  for (int b = 0; b < batch; b++) {
    for (int s = 0; s < seq; s++) {
      for (int h = 0; h < num_heads; h++) {
        const float* in_ptr = src + b * num_heads * seq * d_k + h * seq * d_k + s * d_k;
        float* out_ptr = dst + b * seq * d_model + s * d_model + h * d_k;
        for (int d = 0; d < d_k; d++) {
          out_ptr[d] = in_ptr[d];
        }
      }
    }
  }
}

// SGD update: param[i] -= lr * grad[i]  (uses cblas_saxpy: y += alpha*x)
void tensor_saxpy(int n, float alpha, const float* x, float* y) {
  cblas_saxpy(n, alpha, x, 1, y, 1);
}

// ReLU backward: dx[i] = x[i] > 0 ? dy[i] : 0
void tensor_relu_backward(const float* dy, const float* x, float* dx, int n) {
  for (int i = 0; i < n; i++) {
    dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
  }
}

// Bias add forward: result[i] = x[i] + bias[i % last_dim]
// Row-wise loop avoids modulo overhead
void tensor_bias_add_fwd(const float* x, const float* bias, float* out, int total, int last_dim) {
  int rows = total / last_dim;
  for (int r = 0; r < rows; r++) {
    const float* xr = x + r * last_dim;
    float* outr = out + r * last_dim;
    for (int j = 0; j < last_dim; j++) {
      outr[j] = xr[j] + bias[j];
    }
  }
}

// Bias add backward (dbias): db[j] += sum_rows dy[r, j]
// Row-wise loop avoids modulo overhead
void tensor_bias_add_bwd(const float* dy, float* db, int total, int last_dim) {
  int rows = total / last_dim;
  for (int r = 0; r < rows; r++) {
    cblas_saxpy(last_dim, 1.0f, dy + r * last_dim, 1, db, 1);
  }
}

// Bias add in-place: y[i] += bias[i % last_dim]
void tensor_bias_add_inplace(float* y, const float* bias, int total, int last_dim) {
  int rows = total / last_dim;
  for (int r = 0; r < rows; r++) {
    cblas_saxpy(last_dim, 1.0f, bias, 1, y + r * last_dim, 1);
  }
}

// Scale: x[i] *= s
void tensor_scale_inplace(float* x, float s, int n) {
  cblas_sscal(n, s, x, 1);
}

// ReLU forward: out[i] = max(0, x[i])
void tensor_relu_forward(const float* x, float* out, int n) {
  for (int i = 0; i < n; i++) {
    out[i] = x[i] > 0.0f ? x[i] : 0.0f;
  }
}

// Element-wise accumulate: dst[i] += src[i] (same as saxpy with alpha=1)
void tensor_accumulate(float* dst, const float* src, int n) {
  cblas_saxpy(n, 1.0f, src, 1, dst, 1);
}

// Arena-based ReLU backward: all buffers are offsets into one contiguous arena
void tensor_relu_backward_arena(
  float* arena, int dy_off, int x_off, int dx_off, int n
) {
  const float* dy = arena + dy_off;
  const float* x = arena + x_off;
  float* dx = arena + dx_off;
  for (int i = 0; i < n; i++) {
    dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
  }
}

// Hybrid ReLU forward: x from arena at offset, output to separate buffer
void tensor_relu_forward_hybrid(
  const float* arena, int x_off, float* out, int n
) {
  const float* x = arena + x_off;
  for (int i = 0; i < n; i++) {
    out[i] = x[i] > 0.0f ? x[i] : 0.0f;
  }
}

// Hybrid ReLU backward: dy from separate buffer, x and dx from arena
void tensor_relu_backward_hybrid(
  const float* dy, float* arena, int x_off, int dx_off, int n
) {
  const float* x = arena + x_off;
  float* dx = arena + dx_off;
  for (int i = 0; i < n; i++) {
    dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
  }
}

// Arena-based ReLU forward: x and out are offsets into one contiguous arena
void tensor_relu_forward_arena(
  float* arena, int x_off, int out_off, int n
) {
  const float* x = arena + x_off;
  float* out = arena + out_off;
  for (int i = 0; i < n; i++) {
    out[i] = x[i] > 0.0f ? x[i] : 0.0f;
  }
}

// Arena-based bias add inplace: y is offset into arena
void tensor_bias_add_inplace_arena(
  float* arena, int y_off, const float* bias, int total, int last_dim
) {
  float* y = arena + y_off;
  int rows = total / last_dim;
  for (int r = 0; r < rows; r++) {
    cblas_saxpy(last_dim, 1.0f, bias, 1, y + r * last_dim, 1);
  }
}

// Arena-based bias add backward: dy is offset into arena
void tensor_bias_add_bwd_arena(
  const float* arena, int dy_off, float* db, int total, int last_dim
) {
  const float* dy = arena + dy_off;
  int rows = total / last_dim;
  for (int r = 0; r < rows; r++) {
    cblas_saxpy(last_dim, 1.0f, dy + r * last_dim, 1, db, 1);
  }
}

// Arena-based accumulate: dst and src are offsets into one contiguous arena
void tensor_accumulate_arena(
  float* arena, int dst_off, int src_off, int n
) {
  cblas_saxpy(n, 1.0f, arena + src_off, 1, arena + dst_off, 1);
}

// ========== C-managed ReLU workspace ==========
// Statically managed workspace for pre-relu values.
// Uses posix_memalign for page-aligned memory, completely outside GC control.
// This avoids GC old-generation cache aliasing that causes 10x relu_backward slowdown.

static float* g_relu_pre = NULL;
static int g_relu_pre_cap = 0;

static void ensure_relu_pre(int n) {
  if (g_relu_pre_cap >= n) return;
  free(g_relu_pre);
  // Page-aligned allocation (4096 bytes) for optimal cache behavior
  posix_memalign((void**)&g_relu_pre, 4096, sizeof(float) * n);
  g_relu_pre_cap = n;
}

// Write to C-managed relu_pre buffer (called during forward)
float* tensor_relu_pre_get(int n) {
  ensure_relu_pre(n);
  return g_relu_pre;
}

// sgemm that writes output to C-managed relu_pre buffer
void tensor_sgemm_to_relu_pre(
  int trans_a, int trans_b,
  int m, int n, int k, float alpha,
  const float* a, int lda,
  const float* b, int ldb,
  int total
) {
  ensure_relu_pre(total);
  cblas_sgemm(CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k, alpha, a, lda, b, ldb, 0.0f, g_relu_pre, n);
}

// Bias add inplace on relu_pre
void tensor_bias_add_relu_pre(const float* bias, int total, int last_dim) {
  int rows = total / last_dim;
  for (int r = 0; r < rows; r++) {
    cblas_saxpy(last_dim, 1.0f, bias, 1, g_relu_pre + r * last_dim, 1);
  }
}

// ReLU forward: reads from g_relu_pre, writes to output FixedArray
void tensor_relu_from_pre(float* out, int n) {
  for (int i = 0; i < n; i++) {
    out[i] = g_relu_pre[i] > 0.0f ? g_relu_pre[i] : 0.0f;
  }
}

// ReLU backward: reads dy from MoonBit buffer, reads x from g_relu_pre, writes dx to MoonBit buffer
void tensor_relu_backward_from_pre(const float* dy, float* dx, int n) {
  for (int i = 0; i < n; i++) {
    dx[i] = g_relu_pre[i] > 0.0f ? dy[i] : 0.0f;
  }
}

// ========== C-managed gradient pass-through buffer ==========
// Used by linear backward to write dx, avoiding GC old-gen placement.

static float* g_grad_buf = NULL;
static int g_grad_buf_cap = 0;

static void ensure_grad_buf(int n) {
  if (g_grad_buf_cap >= n) return;
  free(g_grad_buf);
  posix_memalign((void**)&g_grad_buf, 4096, sizeof(float) * n);
  g_grad_buf_cap = n;
}

// sgemm that writes output to C-managed grad buffer
void tensor_sgemm_to_grad_buf(
  int trans_a, int trans_b,
  int m, int n, int k, float alpha,
  const float* a, int lda,
  const float* b, int ldb,
  int total
) {
  ensure_grad_buf(total);
  cblas_sgemm(CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k, alpha, a, lda, b, ldb, 0.0f, g_grad_buf, n);
}

// Copy from C-managed grad buffer to MoonBit FixedArray
void tensor_grad_buf_to_fixed(float* dst, int n) {
  for (int i = 0; i < n; i++) {
    dst[i] = g_grad_buf[i];
  }
}

// ReLU backward: reads dy from g_grad_buf, reads x from g_relu_pre, writes dx to MoonBit buffer
void tensor_relu_backward_managed(float* dx, int n) {
  for (int i = 0; i < n; i++) {
    dx[i] = g_relu_pre[i] > 0.0f ? g_grad_buf[i] : 0.0f;
  }
}

// ========== C-managed relu_dx buffer ==========
// Third C-managed buffer: holds relu backward output (relu_dx).
// Keeps the entire fused backward hot path in C-managed memory,
// avoiding FixedArray::make inside closures that triggers GC + cache pollution.

static float* g_relu_dx_base = NULL;
static float* g_relu_dx = NULL;
static int g_relu_dx_cap = 0;

static void ensure_relu_dx(int n) {
  if (g_relu_dx_cap >= n) return;
  free(g_relu_dx_base);
  // Offset by 16KB from page boundary to avoid cache aliasing with
  // g_relu_pre and g_grad_buf (both page-aligned at offset 0).
  // On Apple M-series, L1 is 128KB 8-way → 256 sets × 64B lines.
  // 16KB offset = 256 cache lines shift, breaking systematic set conflicts.
  size_t alloc_size = sizeof(float) * n + 16384;
  posix_memalign((void**)&g_relu_dx_base, 4096, alloc_size);
  g_relu_dx = (float*)((char*)g_relu_dx_base + 16384);
  g_relu_dx_cap = n;
}

// ReLU backward fully managed: reads dy from g_grad_buf, x from g_relu_pre,
// writes dx to g_relu_dx. All three buffers are C-managed.
void tensor_relu_backward_fully_managed(int n) {
  ensure_relu_dx(n);
  for (int i = 0; i < n; i++) {
    g_relu_dx[i] = g_relu_pre[i] > 0.0f ? g_grad_buf[i] : 0.0f;
  }
}

// Get pointer to g_relu_dx for use in sgemm (as A or B operand)
const float* tensor_relu_dx_ptr(int n) {
  ensure_relu_dx(n);
  return g_relu_dx;
}

// sgemm where A is g_relu_dx: C = alpha * g_relu_dx^T @ B + beta * C
// Used for dW = relu_dx^T @ x
void tensor_sgemm_relu_dx_a(
  int trans_a, int trans_b,
  int m, int n, int k, float alpha,
  int lda,
  const float* b, int ldb,
  float beta, float* c, int ldc
) {
  cblas_sgemm(CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k, alpha, g_relu_dx, lda, b, ldb, beta, c, ldc);
}

// bias_add_backward reading from g_relu_dx: db[j] += sum_rows relu_dx[r, j]
void tensor_bias_add_bwd_relu_dx(float* db, int total, int last_dim) {
  int rows = total / last_dim;
  for (int r = 0; r < rows; r++) {
    cblas_saxpy(last_dim, 1.0f, g_relu_dx + r * last_dim, 1, db, 1);
  }
}

// Copy from g_relu_dx to MoonBit FixedArray (for accumulate_raw if needed)
void tensor_relu_dx_to_fixed(float* dst, int n) {
  for (int i = 0; i < n; i++) {
    dst[i] = g_relu_dx[i];
  }
}

uint64_t timer_clock_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
