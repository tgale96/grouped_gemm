# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import grouped_gemm_backend as backend


def _allocate_output(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b)
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a else
        (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    return torch.empty(*shape, device=a.device, dtype=a.dtype)

def gmm_base(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    backend.gmm_base(a, b, c, batch_sizes, trans_a, trans_b)
    return c

def gmm_cuBLAS(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    backend.gmm_cuBLAS(a, b, c, batch_sizes, trans_a, trans_b)
    return c

def gmm_CUTLASS_sm80(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    backend.gmm_CUTLASS_sm80(a, b, c, batch_sizes, trans_a, trans_b)
    return c

def gmm_CUTLASS_sm90_cooperative(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    backend.gmm_CUTLASS_sm90(a, b, c, batch_sizes, trans_a, trans_b, False)
    return c

def gmm_CUTLASS_sm90_pingpong(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    backend.gmm_CUTLASS_sm90(a, b, c, batch_sizes, trans_a, trans_b, True)
    return c