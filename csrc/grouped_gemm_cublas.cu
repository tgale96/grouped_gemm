#include "grouped_gemm_cublas.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <vector>
#include <numeric>

namespace grouped_gemm {

#define CUDA_CALL(expr)                                                          \
  do {                                                                           \
    cudaError_t _status = (expr);                                                \
    TORCH_CHECK(_status == cudaSuccess,                                          \
                "CUDA Error: ", cudaGetErrorString(_status));                    \
  } while (0)

#define CUBLAS_CALL(expr)                                                        \
  do {                                                                           \
    cublasStatus_t _status = (expr);                                             \
    TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                                \
                "cuBLAS Error: ", static_cast<int>(_status));                    \
  } while (0)

static torch::Tensor make_device_pointer_array(const std::vector<void*>& host,
                                               const torch::Device& device) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
  torch::Tensor t = torch::empty({static_cast<long>(host.size())}, options);

  CUDA_CALL(cudaMemcpyAsync(
      t.data_ptr<int64_t>(),
      host.data(),
      host.size() * sizeof(void*),
      cudaMemcpyHostToDevice,
      at::cuda::getCurrentCUDAStream()));

  return t;
}

static void CublasGroupedGemm_FixedK(torch::Tensor a,
                                     torch::Tensor b,
                                     torch::Tensor c,
                                     torch::Tensor batch_sizes,
                                     bool trans_b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
              "All tensors must be CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 &&
              b.scalar_type() == torch::kBFloat16 &&
              c.scalar_type() == torch::kBFloat16,
              "a, b, c must be bfloat16");
  TORCH_CHECK(a.dim() == 2, "a must be 2D (tokens, hidden_in)");
  TORCH_CHECK(b.dim() == 3, "b must be 3D (num_experts, *, *)");
  TORCH_CHECK(c.dim() == 2, "c must be 2D (tokens, hidden_out)");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "a, b, c must be contiguous");

  TORCH_CHECK(batch_sizes.dim() == 1,
              "batch_sizes must be 1D (num_experts,)");
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64,
              "batch_sizes must be int64");

  const int64_t num_experts = batch_sizes.size(0);
  const int64_t tokens      = a.size(0);
  const int64_t hidden_in   = a.size(1);

  TORCH_CHECK(b.size(0) == num_experts,
              "b.size(0) must equal num_experts");

  const int64_t b_rows = b.size(1);
  const int64_t b_cols = b.size(2);

  const int64_t hidden_out = trans_b ? b_rows : b_cols;

  TORCH_CHECK(hidden_in == (trans_b ? b_cols : b_rows),
              "Incompatible shapes between a and b");
  TORCH_CHECK(c.size(0) == tokens && c.size(1) == hidden_out,
              "c must have shape (tokens, hidden_out)");

  auto batch_sizes_cpu = batch_sizes.to(torch::kCPU);
  const int64_t* bs_ptr = batch_sizes_cpu.data_ptr<int64_t>();

  int64_t tokens_sum = 0;
  for (int64_t i = 0; i < num_experts; ++i) {
    TORCH_CHECK(bs_ptr[i] >= 0, "batch_sizes must be non-negative");
    tokens_sum += bs_ptr[i];
  }
  TORCH_CHECK(tokens_sum == tokens,
              "Sum of batch_sizes must equal total tokens");

  const int m_gemm = static_cast<int>(trans_b ? b_rows : b_cols);
  const int k_gemm = static_cast<int>(trans_b ? b_cols : b_rows);

  TORCH_CHECK(m_gemm > 0 && k_gemm > 0,
              "Invalid GEMM dimensions m or k");

  const bool trans_a = false; 
  const int lda_val = k_gemm;
  const int ldb_val = trans_b ? k_gemm : m_gemm;
  const int ldc_val = static_cast<int>(hidden_out);

  std::vector<cublasOperation_t> transa_array(num_experts);
  std::vector<cublasOperation_t> transb_array(num_experts);
  std::vector<int> m_array(num_experts);
  std::vector<int> n_array(num_experts);
  std::vector<int> k_array(num_experts);
  std::vector<int> lda_array(num_experts);
  std::vector<int> ldb_array(num_experts);
  std::vector<int> ldc_array(num_experts);
  std::vector<int> group_size(num_experts, 1);
  std::vector<float> alpha_array(num_experts, 1.0f);
  std::vector<float> beta_array(num_experts, 0.0f);

  const cublasOperation_t opA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opB = CUBLAS_OP_N;

  for (int64_t i = 0; i < num_experts; ++i) {
    const int64_t tokens_i = bs_ptr[i];
    TORCH_CHECK(tokens_i <= std::numeric_limits<int>::max(),
                "tokens per expert exceed INT32 range");

    transa_array[i] = opA;
    transb_array[i] = opB;
    m_array[i]      = m_gemm;
    n_array[i]      = static_cast<int>(tokens_i);
    k_array[i]      = k_gemm;
    lda_array[i]    = ldb_val;
    ldb_array[i]    = lda_val;
    ldc_array[i]    = ldc_val;
  }

  std::vector<void*> Aarray_host(num_experts);
  std::vector<void*> Barray_host(num_experts);
  std::vector<void*> Carray_host(num_experts);

  const auto device = a.device();
  const auto* a_ptr_base = a.data_ptr<c10::BFloat16>();
  const auto* b_ptr_base = b.data_ptr<c10::BFloat16>();
  auto*       c_ptr_base = c.data_ptr<c10::BFloat16>();

  const int64_t stride_b = b_rows * b_cols;

  int64_t token_offset = 0;
  for (int64_t i = 0; i < num_experts; ++i) {
    const int64_t tokens_i = bs_ptr[i];

    const c10::BFloat16* b_i = b_ptr_base + i * stride_b;
    const c10::BFloat16* a_i = a_ptr_base + token_offset * hidden_in;
    c10::BFloat16*       c_i = c_ptr_base + token_offset * hidden_out;

    Aarray_host[i] = const_cast<c10::BFloat16*>(b_i);
    Barray_host[i] = const_cast<c10::BFloat16*>(a_i);
    Carray_host[i] = c_i;

    token_offset += tokens_i;
  }

  TORCH_CHECK(token_offset == tokens,
              "Internal error: token_offset mismatch");

  torch::Tensor Aarray_dev = make_device_pointer_array(Aarray_host, device);
  torch::Tensor Barray_dev = make_device_pointer_array(Barray_host, device);
  torch::Tensor Carray_dev = make_device_pointer_array(Carray_host, device);

  const void* const* d_Aarray =
      reinterpret_cast<const void* const*>(Aarray_dev.data_ptr<int64_t>());
  const void* const* d_Barray =
      reinterpret_cast<const void* const*>(Barray_dev.data_ptr<int64_t>());
  void* const* d_Carray =
      reinterpret_cast<void* const*>(Carray_dev.data_ptr<int64_t>());

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  const cudaDataType Atype = CUDA_R_16BF;
  const cudaDataType Btype = CUDA_R_16BF;
  const cudaDataType Ctype = CUDA_R_16BF;
  const cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

  CUBLAS_CALL(cublasGemmGroupedBatchedEx(
      handle,
      transa_array.data(),
      transb_array.data(),
      m_array.data(),
      n_array.data(),
      k_array.data(),
      alpha_array.data(),
      d_Aarray,
      Atype,
      lda_array.data(),
      d_Barray,
      Btype,
      ldb_array.data(),
      beta_array.data(),
      d_Carray,
      Ctype,
      ldc_array.data(),
      static_cast<int>(num_experts),
      group_size.data(),
      computeType));
}

void CublasGroupedGemm_VariableK(torch::Tensor a,
                          torch::Tensor b,
                          torch::Tensor c,
                          torch::Tensor batch_sizes) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
              "All tensors must be CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 &&
              b.scalar_type() == torch::kBFloat16 &&
              c.scalar_type() == torch::kBFloat16,
              "a, b, c must be bfloat16");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(b.dim() == 2, "b must be 2D");
  TORCH_CHECK(c.dim() == 3, "c must be 3D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "a, b, c must be contiguous");

  TORCH_CHECK(batch_sizes.dim() == 1,
              "batch_sizes must be 1D (num_experts,)");
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64,
              "batch_sizes must be int64");

  const int64_t num_experts = batch_sizes.size(0);
  const int64_t m = a.size(1);
  const int64_t n = b.size(1);

  TORCH_CHECK(c.size(0) == num_experts &&
              c.size(1) == m &&
              c.size(2) == n,
              "c must have shape (num_experts, m, n)");

  auto batch_sizes_cpu = batch_sizes.to(torch::kCPU);
  const int64_t* k_host = batch_sizes_cpu.data_ptr<int64_t>();

  int64_t sum_k = 0;
  for (int64_t i = 0; i < num_experts; ++i) {
    TORCH_CHECK(k_host[i] >= 0, "batch_sizes must be non-negative");
    sum_k += k_host[i];
  }
  TORCH_CHECK(a.size(0) == sum_k,
              "a.size(0) must equal sum of batch_sizes");
  TORCH_CHECK(b.size(0) == sum_k,
              "b.size(0) must equal sum of batch_sizes");

  const int m_gemm = static_cast<int>(n);
  const int n_gemm = static_cast<int>(m);

  TORCH_CHECK(m_gemm > 0 && n_gemm > 0,
              "Invalid GEMM dimensions m or n");

  std::vector<cublasOperation_t> transa_array(num_experts);
  std::vector<cublasOperation_t> transb_array(num_experts);
  std::vector<int> m_array(num_experts);
  std::vector<int> n_array(num_experts);
  std::vector<int> k_array(num_experts);
  std::vector<int> lda_array(num_experts);
  std::vector<int> ldb_array(num_experts);
  std::vector<int> ldc_array(num_experts);
  std::vector<int> group_size(num_experts, 1);
  std::vector<float> alpha_array(num_experts, 1.0f);
  std::vector<float> beta_array(num_experts, 0.0f);

  const cublasOperation_t opA = CUBLAS_OP_N;
  const cublasOperation_t opB = CUBLAS_OP_T;

  const int lda_val = n_gemm;
  const int ldb_val = m_gemm;
  const int ldc_val = static_cast<int>(n);

  for (int64_t i = 0; i < num_experts; ++i) {
    TORCH_CHECK(k_host[i] <= std::numeric_limits<int>::max(),
                "k_i exceeds INT32 range");

    transa_array[i] = opA;
    transb_array[i] = opB;
    m_array[i]      = m_gemm;
    n_array[i]      = n_gemm;
    k_array[i]      = static_cast<int>(k_host[i]);
    lda_array[i]    = ldb_val;
    ldb_array[i]    = lda_val;
    ldc_array[i]    = ldc_val;
  }

  std::vector<void*> Aarray_host(num_experts);
  std::vector<void*> Barray_host(num_experts);
  std::vector<void*> Carray_host(num_experts);

  const auto device = a.device();
  const auto* a_ptr_base = a.data_ptr<c10::BFloat16>();
  const auto* b_ptr_base = b.data_ptr<c10::BFloat16>();
  auto*       c_ptr_base = c.data_ptr<c10::BFloat16>();

  int64_t offset_a = 0;
  int64_t offset_b = 0;

  const int64_t c_stride = m * n;

  for (int64_t i = 0; i < num_experts; ++i) {
    const int64_t k_i = k_host[i];

    const c10::BFloat16* a_i = a_ptr_base + offset_a;
    const c10::BFloat16* b_i = b_ptr_base + offset_b;
    c10::BFloat16*       c_i = c_ptr_base + i * c_stride;

    Aarray_host[i] = const_cast<c10::BFloat16*>(b_i);
    Barray_host[i] = const_cast<c10::BFloat16*>(a_i);
    Carray_host[i] = c_i;

    offset_a += k_i * m;
    offset_b += k_i * n;
  }

  TORCH_CHECK(offset_a == a.size(0) * m,
              "Internal error: offset_a mismatch");
  TORCH_CHECK(offset_b == b.size(0) * n,
              "Internal error: offset_b mismatch");

  torch::Tensor Aarray_dev = make_device_pointer_array(Aarray_host, device);
  torch::Tensor Barray_dev = make_device_pointer_array(Barray_host, device);
  torch::Tensor Carray_dev = make_device_pointer_array(Carray_host, device);

  const void* const* d_Aarray =
      reinterpret_cast<const void* const*>(Aarray_dev.data_ptr<int64_t>());
  const void* const* d_Barray =
      reinterpret_cast<const void* const*>(Barray_dev.data_ptr<int64_t>());
  void* const* d_Carray =
      reinterpret_cast<void* const*>(Carray_dev.data_ptr<int64_t>());

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  const cudaDataType Atype = CUDA_R_16BF;
  const cudaDataType Btype = CUDA_R_16BF;
  const cudaDataType Ctype = CUDA_R_16BF;
  const cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

  CUBLAS_CALL(cublasGemmGroupedBatchedEx(
      handle,
      transa_array.data(),
      transb_array.data(),
      m_array.data(),
      n_array.data(),
      k_array.data(),
      alpha_array.data(),
      d_Aarray,
      Atype,
      lda_array.data(),
      d_Barray,
      Btype,
      ldb_array.data(),
      beta_array.data(),
      d_Carray,
      Ctype,
      ldc_array.data(),
      static_cast<int>(num_experts),
      group_size.data(),
      computeType));
}

void GroupedGemm_cuBLAS(torch::Tensor a,
                      torch::Tensor b,
                      torch::Tensor c,
                      torch::Tensor batch_sizes,
                      bool trans_a,
                      bool trans_b) {
  TORCH_CHECK(!(trans_a && trans_b),
              "Only one of trans_a / trans_b may be true");

  if (trans_a) {
    CublasGroupedGemm_VariableK(a, b, c, batch_sizes);
  } else {
    CublasGroupedGemm_FixedK(a, b, c, batch_sizes, trans_b);
  }
}

} // namespace grouped_gemm
