#include "grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

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

template <bool trans>
using GroupedGemmInputLayout = std::conditional_t<trans, ::cutlass::layout::ColumnMajor, ::cutlass::layout::RowMajor>;

// TODO(tgale): Update this for SM90 when it's supported by CUTLASS.
template <bool trans_a, bool trans_b>
using GroupedGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
  // A operand.
  ::cutlass::bfloat16_t,
  GroupedGemmInputLayout<trans_a>,
  ::cutlass::ComplexTransform::kNone,
  8,
  // B operand.
  ::cutlass::bfloat16_t,
  GroupedGemmInputLayout<trans_b>,
  ::cutlass::ComplexTransform::kNone,
  8,
  // C operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<128, 128, 32>,
  ::cutlass::gemm::GemmShape<64, 64, 32>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t, 8, float, float>,
  // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
  // This parameter is passed in at present to match the APIs of other kernels. The parameter
  // is unused within the kernel.
  ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
  // TODO(tgale): Experiment with GroupScheduleMode.
  // TODO(tgale): Tune this for SM90.
  4>::GemmKernel;

template <bool trans_a, bool trans_b>
using GemmGrouped = ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernel<trans_a, trans_b>>;

template <bool trans_a, bool trans_b>
std::vector<cutlass::gemm::GemmCoord> MakeProblemSizes(torch::Tensor a, torch::Tensor b, torch::Tensor batch_sizes) {
  const size_t num_experts = batch_sizes.size(0);
  const size_t hidden_in = a.size(1), hidden_out = (trans_a || trans_b) ? b.size(1) : b.size(2);
  std::vector<cutlass::gemm::GemmCoord> problem_sizes(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    int64_t bs = batch_sizes.data_ptr<int64_t>()[i];
    problem_sizes[i] = trans_a
      ? cutlass::gemm::GemmCoord(hidden_in, hidden_out, bs)
      : cutlass::gemm::GemmCoord(bs, hidden_out, hidden_in);
  }
  return problem_sizes;
}

template <typename T>
torch::Tensor CopyToDevice(const std::vector<T> &x, const torch::Device &device) {
  size_t bytes = x.size() * sizeof(T);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
  torch::Tensor out = torch::empty(bytes, options);

  CUDA_CALL(cudaMemcpyAsync(out.data_ptr(),
			    x.data(), bytes,
			    cudaMemcpyHostToDevice,
			    c10::cuda::getCurrentCUDAStream()));
  return out;
}

template <typename T>
static void ReorderArray(T* data, const std::vector<size_t>& indices) {
    // For now, simply create a copy of the data and then copy over to the original.
    std::vector<T> copy(data, data + indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        data[i] = copy.at(indices[i]);
    }
}

template <typename Gemm, bool trans_a, bool trans_b>
typename Gemm::Arguments MakeArguments(torch::Tensor a,
				       torch::Tensor b,
				       torch::Tensor c,
				       torch::Tensor batch_sizes) {
  auto problem_sizes_host = MakeProblemSizes<trans_a, trans_b>(a, b, batch_sizes);

  int64_t num_experts_orig = problem_sizes_host.size();

  // Create the host arrays of leading dimension data and pointer data.
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  std::vector<int64_t> lda_host, ldb_host, ldc_host;
  int64_t elements_a = 0, elements_b = 0, elements_c = 0;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  std::vector<ElementA *> ptr_a_host, ptr_b_host, ptr_c_host;

  lda_host.reserve(num_experts_orig);
  ldb_host.reserve(num_experts_orig);
  ldc_host.reserve(num_experts_orig);

  ptr_a_host.reserve(num_experts_orig);
  ptr_b_host.reserve(num_experts_orig);
  ptr_c_host.reserve(num_experts_orig);

  // CUTLASS doesn't handle problems with `k=0` correctly, see https://github.com/NVIDIA/cutlass/pull/1593.
  // Until a fix is available on the CUTLASS side, handle these problems by ourselves.
  int64_t num_experts = 0;
  for (int i = 0; i < num_experts_orig; ++i) {
    auto problem = problem_sizes_host[i];
    if (problem.k() == 0) {
      CUDA_CALL(cudaMemsetAsync((ElementC*)c.data_ptr() + elements_c,
				0,
				problem.m() * problem.n() * sizeof(ElementC),
				c10::cuda::getCurrentCUDAStream()));
    } else {
      lda_host.push_back(LayoutA::packed({problem.m(), problem.k()}).stride(0));
      ldb_host.push_back(LayoutB::packed({problem.k(), problem.n()}).stride(0));
      ldc_host.push_back(LayoutC::packed({problem.m(), problem.n()}).stride(0));

      ptr_a_host.push_back((ElementA*)a.data_ptr() + elements_a);
      ptr_b_host.push_back((ElementB*)b.data_ptr() + elements_b);
      ptr_c_host.push_back((ElementC*)c.data_ptr() + elements_c);

      problem_sizes_host[num_experts++] = problem;
    }

    elements_a += problem.m() * problem.k();
    elements_b += problem.k() * problem.n();
    elements_c += problem.m() * problem.n();
  }
  problem_sizes_host.resize(num_experts);

  // Calculate the number of threadblocks to use and validate the result.
  // NOTE: This is borrowed from FasterTransformer.
  int threadblock_count = Gemm::sufficient(problem_sizes_host.data(), num_experts);
  if (!threadblock_count) {
    TORCH_CHECK(false, "Grouped GEMM execution not possible with HW");
  }

  // Only sort problems when trans_a = True because only this case K are different
  if (trans_a) {
      std::vector<size_t> indices(num_experts);
      std::iota(indices.begin(), indices.end(), 0);
      std::stable_sort(indices.begin(), indices.end(), [&problem_sizes_host](size_t i, size_t j) {
          return problem_sizes_host[i].k() > problem_sizes_host[j].k();
      });

      ReorderArray(problem_sizes_host.data(), indices);
      ReorderArray(lda_host.data(), indices);
      ReorderArray(ldb_host.data(), indices);
      ReorderArray(ldc_host.data(), indices);
      ReorderArray(ptr_a_host.data(), indices);
      ReorderArray(ptr_b_host.data(), indices);
      ReorderArray(ptr_c_host.data(), indices);
  }

  // Copy the problem sizes, pointers and leading dimension data to the device.
  torch::Tensor lda = CopyToDevice(lda_host, a.device());
  torch::Tensor ldb = CopyToDevice(ldb_host, a.device());
  torch::Tensor ldc = CopyToDevice(ldc_host, a.device());
  torch::Tensor ptr_a = CopyToDevice(ptr_a_host, a.device());
  torch::Tensor ptr_b = CopyToDevice(ptr_b_host, a.device());
  torch::Tensor ptr_c = CopyToDevice(ptr_c_host, a.device());
  torch::Tensor problem_sizes = CopyToDevice(problem_sizes_host, a.device());

  typename Gemm::EpilogueOutputOp::Params epilogue_op(/*alpha=*/1.0f, /*beta=*/0.0f);
  typename Gemm::Arguments arguments((cutlass::gemm::GemmCoord*)problem_sizes.data_ptr(),
  				     (int)num_experts,
  				     (int)threadblock_count,
  				     epilogue_op,
  				     (ElementA**)ptr_a.data_ptr(),
  				     (ElementB**)ptr_b.data_ptr(),
  				     (ElementC**)ptr_c.data_ptr(),
  				     (ElementC**)ptr_c.data_ptr(),
  				     /*lda=*/(int64_t*)lda.data_ptr(),
  				     /*ldb=*/(int64_t*)ldb.data_ptr(),
  				     /*ldc=*/(int64_t*)ldc.data_ptr(),
  				     /*ldd=*/(int64_t*)ldc.data_ptr(),
  				     (cutlass::gemm::GemmCoord*)problem_sizes_host.data());
  return arguments;
}

template <bool trans_a, bool trans_b>
torch::Tensor CutlassGroupedGemm(torch::Tensor a,
				 torch::Tensor b,
				 torch::Tensor c,
				 torch::Tensor batch_sizes) {
  using Gemm = GemmGrouped<trans_a, trans_b>;
  Gemm gemm;

  auto arguments = MakeArguments<Gemm, trans_a, trans_b>(a, b, c, batch_sizes);
  int64_t workspace_size = gemm.get_workspace_size(arguments);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(a.device());
  torch::Tensor workspace = torch::empty(workspace_size, options);

  // Initialize the kernel.
  if(gemm.initialize(arguments, workspace.data_ptr()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
  }

  // Execute the kernel in the current stream.
  if(gemm.run(c10::cuda::getCurrentCUDAStream()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
  }
  return c;
}

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
void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b) {
  // NOTE: We only support 'trans_a' or 'trans_b', not both.
  TORCH_CHECK(!(trans_a && trans_b));

  // We expect the batch_sizes on CPU.
  TORCH_CHECK(batch_sizes.is_cpu());
  TORCH_CHECK(batch_sizes.ndimension() == 1);
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16);

#if !defined(GROUPED_GEMM_CUTLASS)
  if (trans_a) {
    // If we can't use CUTLASS for the transposed cases, defer to the variable 'k' helper using cuBLAS
    // for the rest of the op.
    GroupedGemmVariableK(a, b, c, batch_sizes);
    return;
  }
#endif

  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);

  // The expected shapes of 'b' and 'c' are:
  //   * when 'trans_a' is set: b=(tokens, hidden_out),                 c=(num_experts, hidden_in, hidden_out)
  //   * when 'trans_b' is set: b=(num_experts, hidden_out, hidden_in), c=(tokens, hidden_out)
  //   * otherwise:             b=(num_experts, hidden_in, hidden_out), c=(tokens, hidden
  if (trans_a) {
    TORCH_CHECK(b.ndimension() == 2);
    TORCH_CHECK(c.ndimension() == 3);
    TORCH_CHECK(b.size(0) == a.size(0));
    TORCH_CHECK(c.size(0) == batch_sizes.size(0));
    TORCH_CHECK(c.size(1) == a.size(1));
    TORCH_CHECK(c.size(2) == b.size(1));
  } else {
    TORCH_CHECK(b.ndimension() == 3);
    TORCH_CHECK(c.ndimension() == 2);

    // Validate the contraction dimensions match.
    int64_t tokens = a.size(0), num_experts = b.size(0);
    int64_t hidden_in = trans_b ? b.size(2) : b.size(1);
    int64_t hidden_out = trans_b ? b.size(1) : b.size(2);
    TORCH_CHECK(hidden_in == a.size(1));

    // Validate that we have one size per expert.
    TORCH_CHECK(batch_sizes.size(0) == num_experts);
  }

  // NOTE: We support transposition through the 'trans_b' flag.
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());
  TORCH_CHECK(c.is_contiguous());

#if !defined(GROUPED_GEMM_CUTLASS)
  CublasGroupedGemm(a, b, c, batch_sizes, trans_b);
  return;
#else
  if (trans_a) {
    CutlassGroupedGemm<true, false>(a, b, c, batch_sizes);
    return;
  }
  if (trans_b) {
    CutlassGroupedGemm<false, true>(a, b, c, batch_sizes);
    return;
  }
  CutlassGroupedGemm<false, false>(a, b, c, batch_sizes);
  return;
#endif
}

}  // namespace grouped_gemm
