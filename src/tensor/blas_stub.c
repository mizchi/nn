#ifdef __APPLE__
  #define ACCELERATE_NEW_LAPACK
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

uint64_t timer_clock_ns(void);

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

// Fused batched linear backward:
// dy: [n, out_dim], x: [n, in_dim], w: [in_dim, out_dim]
// dx = dy @ w^T, d_w += x^T @ dy
void tensor_batched_linear_backward(
  const float* dy,
  const float* x,
  const float* w,
  float* d_w,
  float* dx,
  int n,
  int in_dim,
  int out_dim
) {
  // dx = dy @ w^T : [n, out_dim] @ [out_dim, in_dim] -> [n, in_dim]
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasTrans,
    n, in_dim, out_dim,
    1.0f,
    dy, out_dim,
    w, out_dim,
    0.0f,
    dx, in_dim
  );

  // d_w += x^T @ dy : [in_dim, n] @ [n, out_dim] -> [in_dim, out_dim]
  cblas_sgemm(
    CblasRowMajor, CblasTrans, CblasNoTrans,
    in_dim, out_dim, n,
    1.0f,
    x, in_dim,
    dy, out_dim,
    1.0f,
    d_w, out_dim
  );
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

void tensor_softmax_inplace_offset(float* data, int offset, int rows, int cols) {
  tensor_softmax_inplace(data + offset, rows, cols);
}

// Softmax backward over rows:
// out[row, col] = attn[row, col] * (d_attn[row, col] - dot(d_attn[row,*], attn[row,*]))
void tensor_softmax_backward_rows(
  const float* d_attn, const float* attn, float* out, int rows, int cols
) {
  for (int i = 0; i < rows; i++) {
    const float* d_row = d_attn + i * cols;
    const float* a_row = attn + i * cols;
    float* out_row = out + i * cols;
    float dot = 0.0f;
    for (int j = 0; j < cols; j++) {
      dot += d_row[j] * a_row[j];
    }
    for (int j = 0; j < cols; j++) {
      out_row[j] = a_row[j] * (d_row[j] - dot);
    }
  }
}

// Temporary workspace for fused attention head backward.
static float* g_attn_tmp_base = NULL;
static float* g_attn_tmp = NULL;
static int g_attn_tmp_cap = 0;
#define ATTN_TMP_OFFSET_BYTES 320

static void ensure_attn_tmp(int n) {
  if (g_attn_tmp_cap >= n) return;
  free(g_attn_tmp_base);
  size_t alloc_size = sizeof(float) * n + ATTN_TMP_OFFSET_BYTES;
  posix_memalign((void**)&g_attn_tmp_base, 4096, alloc_size);
  g_attn_tmp = (float*)((char*)g_attn_tmp_base + ATTN_TMP_OFFSET_BYTES);
  g_attn_tmp_cap = n;
}

// Fused backward for one attention head:
// d_out, q, k, v are [seq, d_k], attn_w is [seq, seq]
// outputs dq, dk, dv are [seq, d_k]
void tensor_attention_head_backward(
  const float* d_out, int d_out_off,
  const float* q, int q_off,
  const float* k, int k_off,
  const float* v, int v_off,
  const float* attn_w,
  float* dq, int dq_off,
  float* dk, int dk_off,
  float* dv, int dv_off,
  int seq,
  int d_k,
  float scale
) {
  int scores_size = seq * seq;
  ensure_attn_tmp(scores_size * 2);
  float* d_attn = g_attn_tmp;
  float* d_scores = g_attn_tmp + scores_size;

  // d_attn = d_out @ v^T : [seq, d_k] @ [d_k, seq] -> [seq, seq]
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasTrans,
    seq, seq, d_k,
    1.0f,
    d_out + d_out_off, d_k,
    v + v_off, d_k,
    0.0f,
    d_attn, seq
  );

  // dv = attn_w^T @ d_out : [seq, seq]^T @ [seq, d_k] -> [seq, d_k]
  cblas_sgemm(
    CblasRowMajor, CblasTrans, CblasNoTrans,
    seq, d_k, seq,
    1.0f,
    attn_w, seq,
    d_out + d_out_off, d_k,
    0.0f,
    dv + dv_off, d_k
  );

  // d_scores = softmax_backward(d_attn, attn_w)
  tensor_softmax_backward_rows(d_attn, attn_w, d_scores, seq, seq);

  // scale backward
  cblas_sscal(scores_size, scale, d_scores, 1);

  // dq = d_scores @ k : [seq, seq] @ [seq, d_k] -> [seq, d_k]
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    seq, d_k, seq,
    1.0f,
    d_scores, seq,
    k + k_off, d_k,
    0.0f,
    dq + dq_off, d_k
  );

  // dk = d_scores^T @ q : [seq, seq]^T @ [seq, d_k] -> [seq, d_k]
  cblas_sgemm(
    CblasRowMajor, CblasTrans, CblasNoTrans,
    seq, d_k, seq,
    1.0f,
    d_scores, seq,
    q + q_off, d_k,
    0.0f,
    dk + dk_off, d_k
  );
}

// Fused backward for all heads in one call.
// total_heads = batch * num_heads
void tensor_attention_backward_batch(
  const float* d_out,
  const float* q,
  const float* k,
  const float* v,
  const float* attn_w,
  float* dq,
  float* dk,
  float* dv,
  int total_heads,
  int seq,
  int d_k,
  float scale
) {
  int data_stride = seq * d_k;
  int w_stride = seq * seq;
  for (int h = 0; h < total_heads; h++) {
    int data_off = h * data_stride;
    int w_off = h * w_stride;
    tensor_attention_head_backward(
      d_out, data_off,
      q, data_off,
      k, data_off,
      v, data_off,
      attn_w + w_off,
      dq, data_off,
      dk, data_off,
      dv, data_off,
      seq,
      d_k,
      scale
    );
  }
}

// Fused backward for one attention head with explicit leading dimensions.
// d_out/q/k/v and dq/dk/dv each represent [seq, d_k] matrices with row stride ld.
static void tensor_attention_head_backward_ld(
  const float* d_out,
  int d_out_off,
  int d_out_ld,
  const float* q,
  int q_off,
  int q_ld,
  const float* k,
  int k_off,
  int k_ld,
  const float* v,
  int v_off,
  int v_ld,
  const float* attn_w,
  float* dq,
  int dq_off,
  int dq_ld,
  float* dk,
  int dk_off,
  int dk_ld,
  float* dv,
  int dv_off,
  int dv_ld,
  int seq,
  int d_k,
  float scale
) {
  int scores_size = seq * seq;
  ensure_attn_tmp(scores_size * 2);
  float* d_attn = g_attn_tmp;
  float* d_scores = g_attn_tmp + scores_size;

  // d_attn = d_out @ v^T : [seq, d_k] @ [d_k, seq] -> [seq, seq]
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasTrans,
    seq, seq, d_k,
    1.0f,
    d_out + d_out_off, d_out_ld,
    v + v_off, v_ld,
    0.0f,
    d_attn, seq
  );

  // dv = attn_w^T @ d_out : [seq, seq]^T @ [seq, d_k] -> [seq, d_k]
  cblas_sgemm(
    CblasRowMajor, CblasTrans, CblasNoTrans,
    seq, d_k, seq,
    1.0f,
    attn_w, seq,
    d_out + d_out_off, d_out_ld,
    0.0f,
    dv + dv_off, dv_ld
  );

  // d_scores = softmax_backward(d_attn, attn_w)
  tensor_softmax_backward_rows(d_attn, attn_w, d_scores, seq, seq);
  cblas_sscal(scores_size, scale, d_scores, 1);

  // dq = d_scores @ k : [seq, seq] @ [seq, d_k] -> [seq, d_k]
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    seq, d_k, seq,
    1.0f,
    d_scores, seq,
    k + k_off, k_ld,
    0.0f,
    dq + dq_off, dq_ld
  );

  // dk = d_scores^T @ q : [seq, seq]^T @ [seq, d_k] -> [seq, d_k]
  cblas_sgemm(
    CblasRowMajor, CblasTrans, CblasNoTrans,
    seq, d_k, seq,
    1.0f,
    d_scores, seq,
    q + q_off, q_ld,
    0.0f,
    dk + dk_off, dk_ld
  );
}

// Fused backward for interleaved layout:
// d_out/q/k/v, dq/dk/dv are [batch, seq, d_model] (d_model = num_heads * d_k).
// Each head slice is taken as [seq, d_k] with leading dimension d_model.
void tensor_attention_backward_batch_interleaved(
  const float* d_out,
  const float* q,
  const float* k,
  const float* v,
  const float* attn_w,
  float* dq,
  float* dk,
  float* dv,
  int batch,
  int num_heads,
  int seq,
  int d_k,
  float scale
) {
  int d_model = num_heads * d_k;
  int w_stride = seq * seq;
  for (int b = 0; b < batch; b++) {
    int batch_off = b * seq * d_model;
    for (int h = 0; h < num_heads; h++) {
      int head_off = batch_off + h * d_k;
      int w_off = (b * num_heads + h) * w_stride;
      tensor_attention_head_backward_ld(
        d_out, head_off, d_model,
        q, head_off, d_model,
        k, head_off, d_model,
        v, head_off, d_model,
        attn_w + w_off,
        dq, head_off, d_model,
        dk, head_off, d_model,
        dv, head_off, d_model,
        seq,
        d_k,
        scale
      );
    }
  }
}

// Fused forward for interleaved layout:
// q/k/v and out are [batch, seq, d_model] (d_model = num_heads * d_k).
// attn_w is [batch * num_heads, seq, seq] flattened.
static void tensor_attention_forward_batch_interleaved_impl(
  const float* q,
  const float* k,
  const float* v,
  const float* mask,
  float* attn_w,
  float* out,
  int batch,
  int num_heads,
  int seq,
  int d_k,
  float scale
) {
  int d_model = num_heads * d_k;
  int w_stride = seq * seq;
  for (int b = 0; b < batch; b++) {
    int batch_off = b * seq * d_model;
    for (int h = 0; h < num_heads; h++) {
      int head_off = batch_off + h * d_k;
      int w_off = (b * num_heads + h) * w_stride;
      float* scores = attn_w + w_off;

      // scores = Q @ K^T * scale : [seq, d_k] @ [d_k, seq] -> [seq, seq]
      cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        seq, seq, d_k,
        scale,
        q + head_off, d_model,
        k + head_off, d_model,
        0.0f,
        scores, seq
      );

      // Apply causal mask if provided.
      if (mask != NULL) {
        for (int r = 0; r < seq; r++) {
          cblas_saxpy(seq, 1.0f, mask + r * seq, 1, scores + r * seq, 1);
        }
      }

      // softmax(scores)
      tensor_softmax_inplace(scores, seq, seq);

      // out_head = scores @ V : [seq, seq] @ [seq, d_k] -> [seq, d_k]
      cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        seq, d_k, seq,
        1.0f,
        scores, seq,
        v + head_off, d_model,
        0.0f,
        out + head_off, d_model
      );
    }
  }
}

void tensor_attention_forward_batch_interleaved(
  const float* q,
  const float* k,
  const float* v,
  float* attn_w,
  float* out,
  int batch,
  int num_heads,
  int seq,
  int d_k,
  float scale
) {
  tensor_attention_forward_batch_interleaved_impl(
    q, k, v, NULL, attn_w, out, batch, num_heads, seq, d_k, scale
  );
}

void tensor_attention_forward_batch_interleaved_masked(
  const float* q,
  const float* k,
  const float* v,
  const float* mask,
  float* attn_w,
  float* out,
  int batch,
  int num_heads,
  int seq,
  int d_k,
  float scale
) {
  tensor_attention_forward_batch_interleaved_impl(
    q, k, v, mask, attn_w, out, batch, num_heads, seq, d_k, scale
  );
}

// Row-wise matrix add:
// dst[row, col] += add[row, col]
void tensor_add_matrix_inplace(float* dst, const float* add, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    cblas_saxpy(cols, 1.0f, add + r * cols, 1, dst + r * cols, 1);
  }
}

void tensor_add_matrix_inplace_offset(
  float* dst, int dst_off, const float* add, int rows, int cols
) {
  tensor_add_matrix_inplace(dst + dst_off, add, rows, cols);
}

// Cross-entropy forward + backward over rows.
// logits: [rows, cols], labels: [rows]
// returns average loss, writes d_logits = (softmax - one_hot(labels)) / rows
float tensor_cross_entropy_fwd_bwd(
  const float* logits,
  const int* labels,
  float* d_logits,
  int rows,
  int cols
) {
  float loss_sum = 0.0f;
  float inv_rows = 1.0f / (float)rows;
  for (int i = 0; i < rows; i++) {
    const float* row = logits + i * cols;
    float* d_row = d_logits + i * cols;
    int label = labels[i];
    float max_val = row[0];
    for (int j = 1; j < cols; j++) {
      if (row[j] > max_val) max_val = row[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
      float ex = expf(row[j] - max_val);
      d_row[j] = ex;
      sum += ex;
    }
    float inv_sum = 1.0f / sum;
    float log_sum = logf(sum);
    loss_sum += -((row[label] - max_val) - log_sum);
    for (int j = 0; j < cols; j++) {
      d_row[j] = d_row[j] * inv_sum * inv_rows;
    }
    d_row[label] -= inv_rows;
  }
  return loss_sum * inv_rows;
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

// LayerNorm backward + residual add:
// out = add + dx, where dx is LayerNorm backward result.
// Also accumulates d_gamma and d_beta.
void tensor_layer_norm_bwd_add(
  const float* dy, const float* add, const float* x, const float* mean, const float* rstd,
  const float* gamma, float* out, float* d_gamma, float* d_beta,
  int outer, int last_dim
) {
  float inv_dim = 1.0f / (float)last_dim;
  for (int i = 0; i < outer; i++) {
    const float* dy_row = dy + i * last_dim;
    const float* add_row = add + i * last_dim;
    const float* x_row = x + i * last_dim;
    float* out_row = out + i * last_dim;
    float m = mean[i];
    float rs = rstd[i];
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
      float dx = rs * (dy_gamma - sum1 - x_hat * sum2);
      out_row[j] = add_row[j] + dx;
    }
  }
}

// In-place variant of tensor_layer_norm_bwd_add:
// add_inout is read as residual and overwritten with residual + dx.
void tensor_layer_norm_bwd_add_inplace(
  const float* dy, float* add_inout, const float* x, const float* mean, const float* rstd,
  const float* gamma, float* d_gamma, float* d_beta,
  int outer, int last_dim
) {
  tensor_layer_norm_bwd_add(
    dy, add_inout, x, mean, rstd, gamma, add_inout, d_gamma, d_beta, outer, last_dim
  );
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

void tensor_zero(float* dst, int n) {
  memset(dst, 0, (size_t)n * sizeof(float));
}

void tensor_adamw_step(
  float* param, const float* grad, float* m, float* v, int n,
  float lr, float beta1, float beta2, float eps, float weight_decay,
  float bias_c1, float bias_c2
) {
  float one_minus_beta1 = 1.0f - beta1;
  float one_minus_beta2 = 1.0f - beta2;
  float inv_bias_c1 = 1.0f / bias_c1;
  float inv_bias_c2 = 1.0f / bias_c2;
  for (int i = 0; i < n; i++) {
    float gi = grad[i];
    float mi = beta1 * m[i] + one_minus_beta1 * gi;
    float vi = beta2 * v[i] + one_minus_beta2 * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi * inv_bias_c1;
    float v_hat = vi * inv_bias_c2;
    float denom = sqrtf(v_hat) + eps;
    float update = m_hat / denom + weight_decay * param[i];
    param[i] -= lr * update;
  }
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

#define CACHE_ALIGN_BYTES 4096
#define GRAD_BUF_OFFSET_BYTES 64
#define RELU_DX_OFFSET_BYTES 128
#define RELU_OUT_OFFSET_BYTES 192

static float* g_relu_pre = NULL;
static int g_relu_pre_cap = 0;
static float* g_relu_out = NULL;
static float* g_relu_out_base = NULL;
static int g_relu_out_cap = 0;
static void ensure_relu_out(int n);

static void ensure_relu_pre(int n) {
  if (g_relu_pre_cap >= n) return;
  free(g_relu_pre);
  // Page-aligned allocation (4096 bytes) for optimal cache behavior
  posix_memalign((void**)&g_relu_pre, CACHE_ALIGN_BYTES, sizeof(float) * n);
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

// ReLU forward: reads from g_relu_pre, writes to output FixedArray.
// Also caches output in g_relu_out for backward mask access.
void tensor_relu_from_pre(float* out, int n) {
  ensure_relu_out(n);
  for (int i = 0; i < n; i++) {
    float v = g_relu_pre[i] > 0.0f ? g_relu_pre[i] : 0.0f;
    out[i] = v;
    g_relu_out[i] = v;
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

static float* g_grad_buf_base = NULL;
static float* g_grad_buf = NULL;
static int g_grad_buf_cap = 0;

static void ensure_grad_buf(int n) {
  if (g_grad_buf_cap >= n) return;
  free(g_grad_buf_base);
  // Shift by one cache line so g_grad_buf does not alias g_relu_pre set mapping.
  size_t alloc_size = sizeof(float) * n + GRAD_BUF_OFFSET_BYTES;
  posix_memalign((void**)&g_grad_buf_base, CACHE_ALIGN_BYTES, alloc_size);
  g_grad_buf = (float*)((char*)g_grad_buf_base + GRAD_BUF_OFFSET_BYTES);
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
  // Keep relu_dx on a different cache-set offset from both relu_pre and grad_buf.
  size_t alloc_size = sizeof(float) * n + RELU_DX_OFFSET_BYTES;
  posix_memalign((void**)&g_relu_dx_base, CACHE_ALIGN_BYTES, alloc_size);
  g_relu_dx = (float*)((char*)g_relu_dx_base + RELU_DX_OFFSET_BYTES);
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

// ========== C-managed relu output buffer ==========
// Caches relu output in C-managed memory for backward pass.
// Avoids reading GC old-gen workspace buffer during backward sgemm.

static void ensure_relu_out(int n) {
  if (g_relu_out_cap >= n) return;
  free(g_relu_out_base);
  // Separate offset from relu_pre/grad_buf/relu_dx to avoid set conflicts on dW2.
  size_t alloc_size = sizeof(float) * n + RELU_OUT_OFFSET_BYTES;
  posix_memalign((void**)&g_relu_out_base, CACHE_ALIGN_BYTES, alloc_size);
  g_relu_out = (float*)((char*)g_relu_out_base + RELU_OUT_OFFSET_BYTES);
  g_relu_out_cap = n;
}

// ReLU forward with caching: alias of tensor_relu_from_pre.
void tensor_relu_from_pre_cached(float* out, int n) {
  tensor_relu_from_pre(out, n);
}

// ========== Fused FFN backward ==========
// Fused backward for Transformer FFN:
//   dy -> (dW2, db2) -> GELU' -> (dW1, db1, dx)

static float* g_ffn_tmp_base = NULL;
static float* g_ffn_tmp = NULL;
static int g_ffn_tmp_cap = 0;
#define FFN_TMP_OFFSET_BYTES 256

static void ensure_ffn_tmp(int n) {
  if (g_ffn_tmp_cap >= n) return;
  free(g_ffn_tmp_base);
  size_t alloc_size = sizeof(float) * n + FFN_TMP_OFFSET_BYTES;
  posix_memalign((void**)&g_ffn_tmp_base, CACHE_ALIGN_BYTES, alloc_size);
  g_ffn_tmp = (float*)((char*)g_ffn_tmp_base + FFN_TMP_OFFSET_BYTES);
  g_ffn_tmp_cap = n;
}

void tensor_fused_ffn_backward(
    const float* dy,           // [n, d_model]
    const float* ff_post_gelu, // [n, d_ff]
    const float* ff_pre_gelu,  // [n, d_ff]
    const float* ln2_out,      // [n, d_model]
    const float* w2,           // [d_ff, d_model]
    const float* w1,           // [d_model, d_ff]
    float* d_w2,               // [d_ff, d_model], accum
    float* d_b2,               // [d_model], accum
    float* d_w1,               // [d_model, d_ff], accum
    float* d_b1,               // [d_ff], accum
    float* dx,                 // [n, d_model], write
    int n,
    int d_model,
    int d_ff
) {
    const float sqrt_2_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    int total = n * d_ff;
    ensure_ffn_tmp(total);

    // db2 += sum_rows(dy)
    for (int r = 0; r < n; r++) {
        cblas_saxpy(d_model, 1.0f, dy + r * d_model, 1, d_b2, 1);
    }

    // dW2 += ff_post_gelu^T @ dy
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        d_ff, d_model, n,
        1.0f, ff_post_gelu, d_ff, dy, d_model,
        1.0f, d_w2, d_model);

    // d_gelu_out = dy @ W2^T -> g_ffn_tmp
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n, d_ff, d_model,
        1.0f, dy, d_model, w2, d_model,
        0.0f, g_ffn_tmp, d_ff);

    // d_pre_gelu = d_gelu_out * gelu'(ff_pre_gelu) (in-place on g_ffn_tmp)
    for (int i = 0; i < total; i++) {
        float x = ff_pre_gelu[i];
        float inner = sqrt_2_pi * (x + coef * x * x * x);
        float t = tanhf(inner);
        float sech2 = 1.0f - t * t;
        float inner_deriv = sqrt_2_pi * (1.0f + 3.0f * coef * x * x);
        float gelu_deriv = 0.5f * (1.0f + t) + 0.5f * x * sech2 * inner_deriv;
        g_ffn_tmp[i] = g_ffn_tmp[i] * gelu_deriv;
    }

    // db1 += sum_rows(d_pre_gelu)
    for (int r = 0; r < n; r++) {
        cblas_saxpy(d_ff, 1.0f, g_ffn_tmp + r * d_ff, 1, d_b1, 1);
    }

    // dW1 += ln2_out^T @ d_pre_gelu
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        d_model, d_ff, n,
        1.0f, ln2_out, d_model, g_ffn_tmp, d_ff,
        1.0f, d_w1, d_ff);

    // dx = d_pre_gelu @ W1^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n, d_model, d_ff,
        1.0f, g_ffn_tmp, d_ff, w1, d_ff,
        0.0f, dx, d_model);
}

// ========== Fused Two-Layer MLP backward ==========
// Performs entire 2-layer MLP backward in a single C call:
//   linear2 backward → relu backward → linear1 backward
// Uses C-managed g_grad_buf, g_relu_pre, g_relu_dx as intermediates.
// This minimizes GC interaction to a single closure dispatch.

static uint64_t g_fused2_t_dw2 = 0;
static uint64_t g_fused2_t_db2 = 0;
static uint64_t g_fused2_t_da1 = 0;
static uint64_t g_fused2_t_relu = 0;
static uint64_t g_fused2_t_dw1_db1 = 0;

void tensor_fused_two_layer_relu_backward(
    const float* dy,         // [batch, output_dim] upstream gradient
    const float* x_data,     // [batch, input_dim] original input
    const float* w2_data,    // [output_dim, hidden_dim] layer2 weight
    float* dw1,              // [hidden_dim, input_dim] output: layer1 weight grad
    float* db1,              // [hidden_dim] output: layer1 bias grad
    float* dw2,              // [output_dim, hidden_dim] output: layer2 weight grad
    float* db2,              // [output_dim] output: layer2 bias grad
    int batch, int input_dim, int hidden_dim, int output_dim
) {
    int a1_total = batch * hidden_dim;
    uint64_t t0 = timer_clock_ns();

    // --- Linear2 backward ---
    // dW2 = dy^T @ a1 (reads from C-managed g_relu_out, NOT MoonBit old-gen)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        output_dim, hidden_dim, batch,
        1.0f, dy, output_dim, g_relu_out, hidden_dim,
        0.0f, dw2, hidden_dim);
    uint64_t t1 = timer_clock_ns();

    // db2 = sum(dy, axis=0)
    memset(db2, 0, output_dim * sizeof(float));
    for (int r = 0; r < batch; r++) {
        cblas_saxpy(output_dim, 1.0f, dy + r * output_dim, 1, db2, 1);
    }
    uint64_t t2 = timer_clock_ns();

    // d_a1 = dy @ W2 → g_grad_buf
    ensure_grad_buf(a1_total);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        batch, hidden_dim, output_dim,
        1.0f, dy, output_dim, w2_data, hidden_dim,
        0.0f, g_grad_buf, hidden_dim);
    uint64_t t3 = timer_clock_ns();

    // --- ReLU backward ---
    // d_y1 = d_a1 * (relu_pre > 0) → g_relu_dx
    ensure_relu_dx(a1_total);
    for (int i = 0; i < a1_total; i++) {
        g_relu_dx[i] = g_relu_out[i] > 0.0f ? g_grad_buf[i] : 0.0f;
    }
    uint64_t t4 = timer_clock_ns();

    // --- Linear1 backward ---
    // dW1 = d_y1^T @ x
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        hidden_dim, input_dim, batch,
        1.0f, g_relu_dx, hidden_dim, x_data, input_dim,
        0.0f, dw1, input_dim);

    // db1 = sum(d_y1, axis=0)
    memset(db1, 0, hidden_dim * sizeof(float));
    for (int r = 0; r < batch; r++) {
        cblas_saxpy(hidden_dim, 1.0f, g_relu_dx + r * hidden_dim, 1, db1, 1);
    }
    uint64_t t5 = timer_clock_ns();

    g_fused2_t_dw2 = t1 - t0;
    g_fused2_t_db2 = t2 - t1;
    g_fused2_t_da1 = t3 - t2;
    g_fused2_t_relu = t4 - t3;
    g_fused2_t_dw1_db1 = t5 - t4;
}

// Last fused two-layer backward profile:
// [dw2, db2, da1, relu, dw1_db1] in nanoseconds.
void tensor_fused_two_layer_last_profile(uint64_t* out5) {
    out5[0] = g_fused2_t_dw2;
    out5[1] = g_fused2_t_db2;
    out5[2] = g_fused2_t_da1;
    out5[3] = g_fused2_t_relu;
    out5[4] = g_fused2_t_dw1_db1;
}

// ========== Fused Linear+ReLU backward ==========
// Performs entire backward pass in C, avoiding GC interaction between operations.
// Reads from C-managed g_grad_buf (upstream dy) and g_relu_pre (pre-relu values).
// Writes to MoonBit FixedArrays for dw, db, and optionally dx.
void tensor_fused_linear_relu_backward(
    const float* x_data,    // [n_rows, in_features] — input to this linear layer
    const float* w_data,    // [out_features, in_features] — weight matrix
    float* dw_data,         // [out_features, in_features] — output: gradient for weight
    float* db_data,         // [out_features] — output: gradient for bias
    float* dx_data,         // [n_rows, in_features] — output: gradient for input (if compute_dx)
    int n_rows, int in_features, int out_features,
    int compute_dx
) {
    int total = n_rows * out_features;

    // Step 1: relu backward — reads g_grad_buf + g_relu_pre, writes g_relu_dx
    ensure_relu_dx(total);
    for (int i = 0; i < total; i++) {
        g_relu_dx[i] = g_relu_pre[i] > 0.0f ? g_grad_buf[i] : 0.0f;
    }

    // Step 2: dW = relu_dx^T @ x
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        out_features, in_features, n_rows,
        1.0f, g_relu_dx, out_features, x_data, in_features,
        0.0f, dw_data, in_features);

    // Step 3: dbias = sum(relu_dx, axis=0)
    memset(db_data, 0, out_features * sizeof(float));
    for (int r = 0; r < n_rows; r++) {
        cblas_saxpy(out_features, 1.0f, g_relu_dx + r * out_features, 1, db_data, 1);
    }

    // Step 4: dx = relu_dx @ W (if needed)
    if (compute_dx) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n_rows, in_features, out_features,
            1.0f, g_relu_dx, out_features, w_data, in_features,
            0.0f, dx_data, in_features);
    }
}

uint64_t timer_clock_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
