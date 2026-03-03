/*
  Adapted from the CUTLASS example on fp8 grouped gemm on hopper gpus:
  https://github.com/NVIDIA/cutlass/blob/v4.0.0/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu
*/

#include <iostream>
#include <vector>
#include <cfloat>
#include <type_traits>

#include <cuda_runtime.h>

#include <torch/extension.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/layout/matrix.h" 

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

namespace grouped_gemm {

using namespace cute;

// Per-group GEMM problem shape: (M, N, K)
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

// Element types
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

// Layout helper: RowMajor vs ColumnMajor depending on transpose flag
template <bool Transpose>
using GroupedGemmInputLayout = std::conditional_t<Transpose, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
using LayoutD = cutlass::layout::RowMajor;

// 16-byte alignment in elements
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Arch / opclass
using ArchTag       = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Tile configs
struct CooperativeConfig {
  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape        = Shape<_128,_256,_64>;
  using ClusterShape     = Shape<_1,_2,_1>;
};

struct PingpongConfig {
  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape        = Shape<_64,_256,_64>;
  using ClusterShape     = Shape<_1,_2,_1>;
};

template <typename ScheduleConfig, bool TransA, bool TransB>
struct GemmGivenSchedule {
  using TileShape        = typename ScheduleConfig::TileShape;
  using ClusterShape     = typename ScheduleConfig::ClusterShape;
  using KernelSchedule   = typename ScheduleConfig::KernelSchedule;
  using EpilogueSchedule = typename ScheduleConfig::EpilogueSchedule;

  using LayoutA = GroupedGemmInputLayout<TransA>;
  using LayoutB = GroupedGemmInputLayout<TransB>;

  // Collective epilogue: D = alpha * Acc + beta * D
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    void, LayoutD*, 1,        // Bias (disabled via void)
    ElementD, LayoutD*, AlignmentD,    // D
    EpilogueSchedule,
    cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>
  >::CollectiveOp;

  // Collective mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA*, AlignmentA,
    ElementB, LayoutB*, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Concrete kernel types
using Gemm_NN_Cooperative = GemmGivenSchedule<CooperativeConfig, false, false>::Gemm;
using Gemm_NN_Pingpong    = GemmGivenSchedule<PingpongConfig, false, false>::Gemm;

using Gemm_TN_Cooperative = GemmGivenSchedule<CooperativeConfig, true,  false>::Gemm;
using Gemm_TN_Pingpong    = GemmGivenSchedule<PingpongConfig, true,  false>::Gemm;

using Gemm_NT_Cooperative = GemmGivenSchedule<CooperativeConfig, false, true>::Gemm;
using Gemm_NT_Pingpong    = GemmGivenSchedule<PingpongConfig, false, true>::Gemm;


// -----------------------------------------------------------------------------
// Host-side argument preparation
// -----------------------------------------------------------------------------

template <typename GemmT, bool TransB>
struct ArgumentsPreparer {
  using StrideA = typename GemmT::GemmKernel::InternalStrideA;
  using StrideB = typename GemmT::GemmKernel::InternalStrideB;
  using StrideD = typename GemmT::GemmKernel::InternalStrideD;

  // Host metadata
  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideD> stride_D_host;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  std::vector<const ElementA*> ptr_A_host;
  std::vector<const ElementB*> ptr_B_host;
  std::vector<ElementD*>       ptr_D_host;

  int prepare_standard(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& batch_sizes
  ) {
    TORCH_CHECK(a.dim() == 2, "a must be [total_M, K]");
    TORCH_CHECK(b.dim() == 3, "b must be [groups, K, N] or [groups, N, K]");
    TORCH_CHECK(c.dim() == 2, "c must be [total_M, N]");

    int64_t groups64 = batch_sizes.size(0);
    int groups = static_cast<int>(groups64);

    int64_t total_M = a.size(0);
    int64_t K       = a.size(1);
    int64_t N       = c.size(1);

    TORCH_CHECK(c.size(0) == total_M,
                "c must have shape [total_M, N]");

    auto* a_base = reinterpret_cast<ElementA*>(a.data_ptr());
    auto* b_base = reinterpret_cast<ElementB*>(b.data_ptr());
    auto* c_base = reinterpret_cast<ElementD*>(c.data_ptr());

    int64_t a_offset = 0;
    int64_t c_offset = 0;

    stride_A_host.resize(groups);
    stride_B_host.resize(groups);
    stride_D_host.resize(groups);
    problem_sizes_host.resize(groups);
    ptr_A_host.resize(groups);
    ptr_B_host.resize(groups);
    ptr_D_host.resize(groups);

    auto* batch_sizes_ptr = batch_sizes.data_ptr<int64_t>();
    for (int g = 0; g < groups; ++g) {
      int64_t M = batch_sizes_ptr[g];
      if (M == 0) {
        ptr_A_host[g] = reinterpret_cast<const ElementA*>(a.data_ptr());
        ptr_B_host[g] = reinterpret_cast<const ElementB*>(b.data_ptr());
        ptr_D_host[g] = reinterpret_cast<ElementD*>(c.data_ptr());

        stride_A_host[g] = StrideA{};
        stride_B_host[g] = StrideB{};
        stride_D_host[g] = StrideD{};

        problem_sizes_host[g] = {0, 0, 0};
        continue;
      }

      ptr_A_host[g] = a_base + a_offset;
      stride_A_host[g] = cutlass::make_cute_packed_stride(StrideA{}, {int(M), int(K), 1});

      ptr_B_host[g] = b_base + int64_t(g) * K * N;
      stride_B_host[g] = cutlass::make_cute_packed_stride(StrideB{}, {int(N), int(K), 1});

      ptr_D_host[g] = c_base + c_offset;
      stride_D_host[g] = cutlass::make_cute_packed_stride(StrideD{}, {int(M), int(N), 1});

      problem_sizes_host[g] = {int(M), int(N), int(K)};

      a_offset += M * K;
      c_offset += M * N;
    }

    return groups;
  }

  int prepare_dynamicK(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& batch_sizes
  ) {
    TORCH_CHECK(a.dim() == 2, "For dynamicK, a must be [total_K, M]");
    TORCH_CHECK(b.dim() == 2, "For dynamicK, b must be [total_K, N]");
    TORCH_CHECK(c.dim() == 3, "For dynamicK, c must be [groups, M, N]");

    int64_t groups64 = batch_sizes.size(0);
    int groups = static_cast<int>(groups64);

    int64_t total_K = a.size(0);
    int64_t M       = a.size(1);
    int64_t N       = b.size(1);

    TORCH_CHECK(c.size(0) == groups64 &&
                c.size(1) == M &&
                c.size(2) == N,
                "c must be [groups, M, N]");

    auto* a_base = reinterpret_cast<ElementA*>(a.data_ptr());
    auto* b_base = reinterpret_cast<ElementB*>(b.data_ptr());
    auto* c_base = reinterpret_cast<ElementD*>(c.data_ptr());

    int64_t K_offset = 0;

    stride_A_host.resize(groups);
    stride_B_host.resize(groups);
    stride_D_host.resize(groups);
    problem_sizes_host.resize(groups);
    ptr_A_host.resize(groups);
    ptr_B_host.resize(groups);
    ptr_D_host.resize(groups);

    auto* batch_sizes_ptr = batch_sizes.data_ptr<int64_t>();
    for (int g = 0; g < groups; ++g) {
      int64_t K = batch_sizes_ptr[g];
      if (K == 0) {
        ptr_A_host[g] = reinterpret_cast<const ElementA*>(a.data_ptr());
        ptr_B_host[g] = reinterpret_cast<const ElementB*>(b.data_ptr());
        ptr_D_host[g] = reinterpret_cast<ElementD*>(c.data_ptr());

        stride_A_host[g] = StrideA{};
        stride_B_host[g] = StrideB{};
        stride_D_host[g] = StrideD{};

        problem_sizes_host[g] = {0, 0, 0};
        continue;
      }

      ptr_A_host[g] = a_base + K_offset * M;
      stride_A_host[g] = cutlass::make_cute_packed_stride(StrideA{}, {int(M), int(K), 1});

      ptr_B_host[g] = b_base + K_offset * N;
      stride_B_host[g] = cutlass::make_cute_packed_stride(StrideB{}, {int(N), int(K), 1});

      ptr_D_host[g] = c_base + int64_t(g) * M * N;
      stride_D_host[g] = cutlass::make_cute_packed_stride(StrideD{}, {int(M), int(N), 1});

      problem_sizes_host[g] = {int(M), int(N), int(K)};

      K_offset += K;
    }

    TORCH_CHECK(K_offset == total_K, "Sum(batch_sizes) (", K_offset, ") must equal total_K (", total_K, ")");

    return groups;
  }
};

template <typename GemmT, bool TransA, bool TransB>
void run(
  const torch::Tensor& a,
  const torch::Tensor& b,
  const torch::Tensor& c,
  const torch::Tensor& batch_sizes,
  int device_id
) {
  GemmT gemm;
  ArgumentsPreparer<GemmT, TransB> prep;

  // Prepare host metadata
  int groups = 0;
  if (TransA) {
    groups = prep.prepare_dynamicK(a, b, c, batch_sizes);
  } else {
    groups = prep.prepare_standard(a, b, c, batch_sizes);
  }

  // Device allocations for per-group metadata
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes_dev;
  cutlass::DeviceAllocation<const typename GemmT::ElementA*> ptr_A_dev;
  cutlass::DeviceAllocation<const typename GemmT::ElementB*> ptr_B_dev;
  cutlass::DeviceAllocation<typename GemmT::EpilogueOutputOp::ElementOutput*> ptr_D_dev;
  cutlass::DeviceAllocation<typename GemmT::GemmKernel::InternalStrideA> stride_A_dev;
  cutlass::DeviceAllocation<typename GemmT::GemmKernel::InternalStrideB> stride_B_dev;
  cutlass::DeviceAllocation<typename GemmT::GemmKernel::InternalStrideD> stride_D_dev;

  problem_sizes_dev.reset(groups);
  problem_sizes_dev.copy_from_host(prep.problem_sizes_host.data());

  ptr_A_dev.reset(groups);
  ptr_A_dev.copy_from_host(prep.ptr_A_host.data());

  ptr_B_dev.reset(groups);
  ptr_B_dev.copy_from_host(prep.ptr_B_host.data());

  ptr_D_dev.reset(groups);
  ptr_D_dev.copy_from_host(prep.ptr_D_host.data());

  stride_A_dev.reset(groups);
  stride_A_dev.copy_from_host(prep.stride_A_host.data());

  stride_B_dev.reset(groups);
  stride_B_dev.copy_from_host(prep.stride_B_host.data());

  stride_D_dev.reset(groups);
  stride_D_dev.copy_from_host(prep.stride_D_host.data());

  // Hardware info
  cutlass::KernelHardwareInfo kernel_hw_info =
    cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename GemmT::GemmKernel>(device_id);

  // Build arguments
  typename GemmT::Arguments arguments{};

  // Value-init thread epilogue params, then override alpha/beta
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta  = 0.0f;
  fusion_args.alpha_ptr       = nullptr;
  fusion_args.beta_ptr        = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array  = nullptr;
  fusion_args.dAlpha          = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta           = {cute::_0{}, cute::_0{}, 0};

  arguments = typename GemmT::Arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {groups, problem_sizes_dev.get(), prep.problem_sizes_host.data()},
    {ptr_A_dev.get(), stride_A_dev.get(), ptr_B_dev.get(), stride_B_dev.get()},
    {fusion_args, nullptr, nullptr, ptr_D_dev.get(), stride_D_dev.get()},
    kernel_hw_info
  };

  // Workspace & run
  size_t workspace_size = GemmT::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
}

#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

void GroupedGemm_CUTLASS_sm90(
  torch::Tensor a,
  torch::Tensor b,
  torch::Tensor c,
  torch::Tensor batch_sizes,
  bool trans_a,
  bool trans_b,
  bool use_pingpong
) {
#if !defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  std::cerr << "CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED not defined. "
               "Compile with CUDA 12.3+ and SM90 targets.\n";
  return;
#else
  // CUTLASS must be compiled with CUDA 12.3 Toolkit to run this function
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "GroupedGemm_CUTLASS_sm90 requires CUDA 12.3 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return;
  }

  // NOTE: We only support 'trans_a' or 'trans_b', not both.
  TORCH_CHECK(!(trans_a && trans_b));

  // CUTLASS can handle both CPU- and CUDA-resident problem dimensions.
  torch::Tensor batch_sizes_cpu = batch_sizes;
  if (!batch_sizes.is_cpu()) {
      batch_sizes_cpu = batch_sizes.to(torch::kCPU);
  }
  batch_sizes_cpu = batch_sizes_cpu.contiguous();
  TORCH_CHECK(batch_sizes_cpu.ndimension() == 1);
  TORCH_CHECK(batch_sizes_cpu.scalar_type() == torch::kInt64);

  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16);

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
    TORCH_CHECK(c.size(0) == batch_sizes_cpu.size(0));
    TORCH_CHECK(c.size(1) == hidden_in);
    TORCH_CHECK(c.size(2) == hidden_out);

    auto c_view = c.view({batch_sizes_cpu.size(0), (long)hidden_in, (long)hidden_out});

    for (int64_t g = 0; g < batch_sizes_cpu.size(0); ++g) {
      if (batch_sizes_cpu[g].item<int64_t>() == 0) {
        c_view[g].zero_();
      }
    }
  } else { //trans_a == false
    TORCH_CHECK(b.ndimension() == 3);
    TORCH_CHECK(c.ndimension() == 2);

    // Validate the contraction dimensions match.
    int64_t tokens = a.size(0), num_experts = b.size(0);
    hidden_in = trans_b ? b.size(2) : b.size(1);
    hidden_out = trans_b ? b.size(1) : b.size(2);
    TORCH_CHECK(hidden_in == a.size(1));

    // Validate that we have one size per expert.
    TORCH_CHECK(batch_sizes_cpu.size(0) == num_experts);
  }

  // NOTE: We support transposition through the 'trans_b' flag.
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());
  TORCH_CHECK(c.is_contiguous());

  // Make sure all inputs are on the same device
  int device_id = a.get_device();
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
  if (props.major != 9 || props.minor != 0) {
    std::cerr
      << "GroupedGemm_CUTLASS_sm90 requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
    return;
  }
  TORCH_CHECK(b.get_device() == device_id, "a and b must be on the same device");
  TORCH_CHECK(c.get_device() == device_id, "a and c must be on the same device");

  // Dispatch on (trans_a, trans_b, use_pingpong)
  if ((!use_pingpong) && (!trans_a) && (!trans_b)) {
    run<Gemm_NN_Cooperative, false, false>(a, b, c, batch_sizes_cpu, device_id);
  } else if ((use_pingpong) && (!trans_a) && (!trans_b)) {
    run<Gemm_NN_Pingpong, false, false>(a, b, c, batch_sizes_cpu, device_id);
  } else if ((!use_pingpong) && (trans_a) && (!trans_b)) {
    run<Gemm_TN_Cooperative, true, false>(a, b, c, batch_sizes_cpu, device_id);
  } else if ((use_pingpong) && (trans_a) && (!trans_b)) {
    run<Gemm_TN_Pingpong, true, false>(a, b, c, batch_sizes_cpu, device_id);
  } else if ((!use_pingpong) && (!trans_a) && (trans_b)) {
    run<Gemm_NT_Cooperative, false, true>(a, b, c, batch_sizes_cpu, device_id);
  } else if ((use_pingpong) && (!trans_a) && (trans_b)) {
    run<Gemm_NT_Pingpong, false, true>(a, b, c, batch_sizes_cpu, device_id);
  } else {
    TORCH_CHECK(false, "GEMM option (trans_a, trans_b, use_pingpong) not supported.");
  }
#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
return;
}

} // namespace grouped_gemm
