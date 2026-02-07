#ifdef __APPLE__
  #define ACCELERATE_NEW_LAPACK
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

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
