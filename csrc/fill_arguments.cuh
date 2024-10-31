#pragma once

#include <ATen/cuda/detail/KernelUtils.h>
#include <cub/cub.cuh>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm_coord.h>

namespace grouped_gemm {

constexpr int kDynamicDim = -1;
constexpr int kMaxExperts = 512;

struct GemmProblem {
  ::cutlass::gemm::GemmCoord dims;
  int64_t lda, ldb, ldc;
  // All offsets are in elements.
  int64_t a_offset, b_offset, c_offset;
};

struct ExtractGemmProblemK {
  __device__ ::cuda::std::tuple<int&> operator()(GemmProblem& problem) const {
      return {problem.dims.k()};
  }
};

template <
    // If `k` is dynamic, we sort the problems by `k` in descending order.
    // Otherwise, `m` is dynamic, and no sorting happens.
    bool kDynamicK,
    typename ElementA, typename ElementB, typename ElementC,
    typename LayoutA, typename LayoutB, typename LayoutC,
    typename Args
>
__global__ void FillArguments(
    int num_experts, const int64_t* batch_sizes,
    ElementA* ptr_a, ElementB* ptr_b, ElementC* ptr_c,
    Args args, ::cutlass::gemm::GemmCoord dims
) {
  const int expert_idx = threadIdx.x;
  const int batch_size = expert_idx < num_experts ? batch_sizes[expert_idx] : -1;

  if (kDynamicK) {
    assert(dims.k() == kDynamicDim);
    dims.k() = batch_size;
  } else {
    assert(dims.m() == kDynamicDim);
    dims.m() = batch_size;
  }

  using BlockScan = cub::BlockScan<int, kMaxExperts>;
  using BlockSort = cub::BlockRadixSort<GemmProblem, kMaxExperts, 1>;

  union SharedMemory {
    BlockScan::TempStorage scan_storage;
    BlockSort::TempStorage sort_storage;
  };
  __shared__ SharedMemory shared_memory;

  int dynamic_dim = kDynamicK ? dims.k() : dims.m();
  int dynamic_dim_cumsum;
  BlockScan(shared_memory.scan_storage).ExclusiveSum(dynamic_dim, dynamic_dim_cumsum);
  __syncthreads();

  // We have to use `GemmProblem[1]` here instead of just `GemmProblem` because `SortDescending()` expects
  // `KeyT (&)[ITEMS_PER_THREAD]` for the `keys` argument (i.e., `GemmProblem (&keys)[1]` in our case).
  GemmProblem problem[1] = {
    GemmProblem {
      .dims = dims,
      .lda = LayoutA::packed({dims.m(), dims.k()}).stride(0),
      .ldb = LayoutB::packed({dims.k(), dims.n()}).stride(0),
      .ldc = LayoutC::packed({dims.m(), dims.n()}).stride(0),
      .a_offset = kDynamicK
          ? (dims.m() * dynamic_dim_cumsum)
          : (dynamic_dim_cumsum * dims.k()),
      .b_offset = (kDynamicK ? dynamic_dim_cumsum : expert_idx * dims.k()) * dims.n(),
      .c_offset = (kDynamicK ? expert_idx * dims.m() : dynamic_dim_cumsum) * dims.n(),
    },
  };

  if constexpr (kDynamicK) {
    BlockSort(shared_memory.sort_storage).SortDescending(problem, ExtractGemmProblemK{});
    // Quoting the CUB documentation (https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockRadixSort.html):
    // > A subsequent __syncthreads() threadblock barrier should be invoked after calling this method if the collective’s temporary storage [...]
    // > is **to be reused or repurposed**.
    // We don't need `__syncthreads()` here, since we don't do either of these things.
  }

  if (expert_idx < num_experts) {
    args.problem_sizes[expert_idx] = problem[0].dims;
    args.lda[expert_idx] = problem[0].lda;
    args.ldb[expert_idx] = problem[0].ldb;
    args.ldc[expert_idx] = problem[0].ldc;

    args.ptr_A[expert_idx] = ptr_a + problem[0].a_offset;
    args.ptr_B[expert_idx] = ptr_b + problem[0].b_offset;
    args.ptr_C[expert_idx] = ptr_c + problem[0].c_offset;
  }
}

template <typename Args>
__global__ void ZeroOutK0Outputs(int num_experts, Args args) {
  const int64_t start_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t delta     = (int64_t)gridDim.x * blockDim.x;
  for (int ei = 0; ei < num_experts; ++ei) {
    auto& dims = args.problem_sizes[ei];
    // CUTLASS doesn't handle problems with `k=0` correctly, see https://github.com/NVIDIA/cutlass/pull/1593.
    // Until a fix is available on the CUTLASS side, handle these problems by ourselves:
    //   * (here) set the output to zero
    //   * (in `IgnoreK0Problems`) make this problem a no-op by setting `m=0` and `n=0` (CUTLASS can handle the outer dimensions being zero)
    if (dims.k() == 0) {
      // Assume packed layout, run a grid-strided loop over the output.
      int64_t total_elems = (int64_t)dims.m() * dims.n();
      auto* out           = args.ptr_C[ei];
      for (int64_t idx = start_idx; idx < total_elems; idx += delta) {
        out[idx] = {};
      }
    }
  }
}

template <typename Args>
__global__ void IgnoreK0Problems(int num_experts, Args args) {
  const int expert_idx = threadIdx.x;
  if (expert_idx < num_experts) {
    auto& dims = args.problem_sizes[expert_idx];
    if (dims.k() == 0) {
      dims.m() = 0;
      dims.n() = 0;
    }
  }
}

}  // namespace grouped_gemm
