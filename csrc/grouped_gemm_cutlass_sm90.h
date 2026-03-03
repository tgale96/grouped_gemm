#include <torch/extension.h>

namespace grouped_gemm {

void GroupedGemm_CUTLASS_sm90(
		 torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, 
		 bool trans_b,
		 bool use_pingpong);

}  // namespace grouped_gemm
