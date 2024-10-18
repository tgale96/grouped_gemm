import unittest
import itertools

from absl.testing import parameterized
from grouped_gemm import ops
import numpy as np
import torch


def allclose(x, y, pct=2.0):
    mask = torch.isclose(x, y, rtol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


def add_flags(x):
    out = []
    for y in x:
        for trans_b in (False, True):
            for batch_sizes_on_device in (False, True):
                out.append(y + (trans_b, batch_sizes_on_device))
    return out


_TEST_PROBLEMS = add_flags((
    (1, 128, 128, 128),
    (8, 128, 128, 128),
    (16, 128, 128, 128),
    (1, 128, 256, 512),
    (8, 128, 256, 512),
    (16, 128, 256, 512),
))


def randn(bs, x, y):
    out = (torch.rand(bs, x, y) - 0.5 * 2) / (y * x)
    return out.cuda().to(torch.bfloat16)


def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.cpu().numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)


@parameterized.parameters(*_TEST_PROBLEMS)
class OpsTest(parameterized.TestCase):

    def testGroupedGemm_FixedSizes(self, z, m, k, n, trans_b, batch_sizes_on_device):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)
        batch_sizes = torch.tensor([m] * z)
        if batch_sizes_on_device:
            batch_sizes = batch_sizes.cuda()

        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes, trans_b)
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        # Check gradients.
        out.sum().backward()
        expected_out.sum().backward()
        self.assertTrue(allclose(a.grad, a_ref.grad))
        self.assertTrue(allclose(b.grad, b_ref.grad))

    def testGroupedGemm_VariableSizes(self, z, m, k, n, trans_b, batch_sizes_on_device):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)

        dist = torch.rand(z, )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)
        if batch_sizes_on_device:
            batch_sizes = batch_sizes.cuda()

        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes, trans_b)
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        # Check gradients.
        out.sum().backward()
        expected_out.sum().backward()
        self.assertTrue(allclose(a.grad, a_ref.grad))
        self.assertTrue(allclose(b.grad, b_ref.grad))


@parameterized.parameters(False, True)
class EdgeCasesTest(unittest.TestCase):

    def testGroupedGemm_ZeroSize(self, batch_sizes_on_device):
        torch.manual_seed(0)
        m = 16384
        k = 4096
        n = 14336
        num_experts = 8

        a = randn(num_experts, m // num_experts, k).view(-1, k)
        b = randn(num_experts, k, n)
        batch_sizes = torch.tensor([219, 2246, 5, 8103, 1, 1117, 4693, 0]).to(torch.long)
        if batch_sizes_on_device:
            batch_sizes = batch_sizes.cuda()

        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes)
        expected_out = gmm(a_ref, b_ref, batch_sizes)
        self.assertTrue(allclose(out, expected_out))

        # Check gradients.
        out.sum().backward()
        expected_out.sum().backward()
        self.assertTrue(allclose(a.grad, a_ref.grad))
        self.assertTrue(allclose(b.grad, b_ref.grad))

    def testGroupedGemm_ZeroK(self, batch_sizes_on_device):
        sz = 128
        total_tokens = 192

        a = torch.ones(total_tokens, sz).cuda().to(torch.bfloat16)
        b = torch.ones(total_tokens, sz).cuda().to(torch.bfloat16)
        c = torch.ones(4, sz, sz).cuda().to(torch.bfloat16)
        batch_sizes = torch.tensor([0, 128, 0, 64]).to(torch.long)
        if batch_sizes_on_device:
            batch_sizes = batch_sizes.cuda()

        ops.backend.gmm(a, b, batch_sizes, trans_a=True, c=c)
        self.assertTrue((c[0] == 0).all())
        self.assertTrue((c[1] == 128).all())
        self.assertTrue((c[2] == 0).all())
        self.assertTrue((c[3] == 64).all())


if __name__ == '__main__':
    unittest.main()
