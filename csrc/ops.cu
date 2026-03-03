#include "grouped_gemm.h"
#include "grouped_gemm_cublas.h"
#include "grouped_gemm_cutlass_sm80.h"
#include "grouped_gemm_cutlass_sm90.h"

#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm_base", &GroupedGemm_base, "Grouped GEMM base.");
  m.def("gmm_cuBLAS", &GroupedGemm_cuBLAS, "Grouped GEMM cuBLAS.");
  m.def("gmm_CUTLASS_sm80", &GroupedGemm_CUTLASS_sm80, "Grouped GEMM CUTLASS for Ampere (sm80).");
  m.def("gmm_CUTLASS_sm90", &GroupedGemm_CUTLASS_sm90, "Grouped GEMM CUTLASS for Hopper (sm90).");
}

}  // namespace grouped_gemm
