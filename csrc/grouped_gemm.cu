#include "grouped_gemm.h"
#include "fill_arguments.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include <type_traits>

namespace grouped_gemm {

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

#define CUBLAS_CALL(code)					  \
  do {								  \
    cublasStatus_t status = code;				  \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS Error"); \
  } while (0)

#define GROUPED_GEMM_STRINGIFY_HELPER(x) #x
#define GROUPED_GEMM_STRINGIFY(x) \
  GROUPED_GEMM_STRINGIFY_HELPER(x)

void CublasGemm(c10::BFloat16 *a, int64_t a_rows, int64_t a_cols, bool trans_a,
		c10::BFloat16 *b, int64_t b_rows, int64_t b_cols, bool trans_b,
		c10::BFloat16 *c, int64_t c_rows, int64_t c_cols) {
  int m = trans_b ? b_rows : b_cols;
  int k = trans_b ? b_cols : b_rows;
  int n = trans_a ? a_cols : a_rows;

  int lda = trans_a ? n : k;
  int ldb = trans_b ? k : m;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  float alpha = 1.0, beta = 0.0;
  CUBLAS_CALL(cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(),
			   transpose_b, transpose_a,
			   m, n, k, &alpha,
			   b, CUDA_R_16BF, ldb,
			   a, CUDA_R_16BF, lda,
			   &beta,
			   c, CUDA_R_16BF, c_cols, CUDA_R_32F,
			   CUBLAS_GEMM_DEFAULT));
}

void CublasGroupedGemm(torch::Tensor a,
		       torch::Tensor b,
		       torch::Tensor c,
		       torch::Tensor batch_sizes,
		       bool trans_b) {
  int64_t bs = batch_sizes.size(0), k = a.size(1);
  int64_t n = trans_b ? b.size(1) : b.size(2);
  int64_t b_rows = b.size(1), b_cols = b.size(2);
  c10::BFloat16* a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16* b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16* c_ptr = c.data_ptr<c10::BFloat16>();
  for (int i = 0; i < bs; ++i) {
    int64_t m = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(a_ptr, m, k, /*trans_a=*/false,
	       b_ptr, b_rows, b_cols, trans_b,
	       c_ptr, m, n);
    a_ptr += m * k;
    b_ptr += b_rows * b_cols;
    c_ptr += m * n;
  }
}

void CublasGroupedGemmVariableK(torch::Tensor a,
				torch::Tensor b,
				torch::Tensor c,
				torch::Tensor batch_sizes) {
  int64_t bs = batch_sizes.size(0), m = a.size(1), n = b.size(1);
  c10::BFloat16* a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16* b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16* c_ptr = c.data_ptr<c10::BFloat16>();
  for (int i = 0; i < bs; ++i) {
    int64_t k = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(a_ptr, k, m, /*trans_a=*/true,
	       b_ptr, k, n, /*trans_b=*/false,
	       c_ptr, m, n);
    a_ptr += k * m;
    b_ptr += k * n;
    c_ptr += m * n;
  }
}

void GroupedGemmVariableK(torch::Tensor a,
			  torch::Tensor b,
			  torch::Tensor c,
			  torch::Tensor batch_sizes) {
  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 2);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);

  // Validate the dimensions.
  int64_t tokens = a.size(0), num_experts = batch_sizes.size(0);
  int64_t m = a.size(1), n = b.size(1);

  // Validate that we have the same contraction dimension.
  TORCH_CHECK(tokens == b.size(0));

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 3);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == num_experts);
  TORCH_CHECK(c.size(1) == m);
  TORCH_CHECK(c.size(2) == n);

  // Run the computation.
  CublasGroupedGemmVariableK(a, b, c, batch_sizes);
}

// NOTE: We only support dynamic group sizes for the 'a' tensor. Tensor 'b' is
// assumed to be batched with fixed sized batches.
//
// TODO(tgale): Validate alignment is true for every batch element.
void GroupedGemm_base(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b) {
  // NOTE: We only support 'trans_a' or 'trans_b', not both.
  TORCH_CHECK(!(trans_a && trans_b));

  // No way to run cuBLAS kernels if the problem dimensions are not known on the host.
  TORCH_CHECK(batch_sizes.is_cpu());
  TORCH_CHECK(batch_sizes.ndimension() == 1);
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16);

  if (trans_a) {
    GroupedGemmVariableK(a, b, c, batch_sizes);
    return;
  }

  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);

  // The expected shapes of 'b' and 'c' are:
  //   * when 'trans_a' is set: b=(tokens, hidden_out),                 c=(num_experts, hidden_in, hidden_out)
  //   * when 'trans_b' is set: b=(num_experts, hidden_out, hidden_in), c=(tokens, hidden_out)
  //   * otherwise:             b=(num_experts, hidden_in, hidden_out), c=(tokens, hidden
  size_t hidden_in{}, hidden_out{};
  if (trans_a) {
    hidden_in = a.size(1);
    hidden_out = b.size(1);

    TORCH_CHECK(b.ndimension() == 2);
    TORCH_CHECK(c.ndimension() == 3);
    TORCH_CHECK(b.size(0) == a.size(0));
    TORCH_CHECK(c.size(0) == batch_sizes.size(0));
    TORCH_CHECK(c.size(1) == hidden_in);
    TORCH_CHECK(c.size(2) == hidden_out);
  } else {
    TORCH_CHECK(b.ndimension() == 3);
    TORCH_CHECK(c.ndimension() == 2);

    // Validate the contraction dimensions match.
    int64_t tokens = a.size(0), num_experts = b.size(0);
    hidden_in = trans_b ? b.size(2) : b.size(1);
    hidden_out = trans_b ? b.size(1) : b.size(2);
    TORCH_CHECK(hidden_in == a.size(1));

    // Validate that we have one size per expert.
    TORCH_CHECK(batch_sizes.size(0) == num_experts);
  }

  // NOTE: We support transposition through the 'trans_b' flag.
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());
  TORCH_CHECK(c.is_contiguous());

  CublasGroupedGemm(a, b, c, batch_sizes, trans_b);
  return;
}

}  // namespace grouped_gemm
