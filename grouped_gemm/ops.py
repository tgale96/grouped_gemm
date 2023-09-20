# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import grouped_gemm_backend as backend

class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes):
        return backend.grouped_gemm(a, b, batch_sizes)
grouped_gemm = GroupedGemm.apply
