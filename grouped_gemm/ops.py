from grouped_gemm import backend
import torch


class GroupedGemmTemplate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b, gemm_op):
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        ctx.gemm_op = gemm_op
        
        return gemm_op(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b
        gemm_op = ctx.gemm_op

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = gemm_op(grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = gemm_op(lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        
        return agrad, bgrad, None, None, None

def gmm_base(a, b, batch_sizes, trans_b=False):
    return GroupedGemmTemplate.apply(a, b, batch_sizes, trans_b, backend.gmm_base)

def gmm_cuBLAS(a, b, batch_sizes, trans_b=False):
    return GroupedGemmTemplate.apply(a, b, batch_sizes, trans_b, backend.gmm_cuBLAS)

def gmm_CUTLASS_sm80(a, b, batch_sizes, trans_b=False):
    return GroupedGemmTemplate.apply(a, b, batch_sizes, trans_b, backend.gmm_CUTLASS_sm80)

def gmm_CUTLASS_sm90_cooperative(a, b, batch_sizes, trans_b=False):
    return GroupedGemmTemplate.apply(a, b, batch_sizes, trans_b, backend.gmm_CUTLASS_sm90_cooperative)

def gmm_CUTLASS_sm90_pingpong(a, b, batch_sizes, trans_b=False):
    return GroupedGemmTemplate.apply(a, b, batch_sizes, trans_b, backend.gmm_CUTLASS_sm90_pingpong)