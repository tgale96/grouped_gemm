import unittest
import itertools

from absl.testing import parameterized
from grouped_gemm import ops
import numpy as np
import torch


def allclose(x, y, pct=0.25):
    mask = torch.isclose(x, y, rtol=1e-3)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


def add_transpose_flags(x):
    out = []
    for y in x:
        for f in [(False,), (True,)]:
            out.append(y + f)
    return out


_TEST_PROBLEMS = add_transpose_flags((
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
    batch_sizes = batch_sizes.numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)


@parameterized.parameters(*_TEST_PROBLEMS)
class OpsTest(parameterized.TestCase):

    def testGroupedGemm_FixedSizes(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)
        batch_sizes = torch.tensor([m] * z)

        out = ops.gmm(a, b, batch_sizes, trans_b)
        expected_out = gmm(a, b, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

    def testGroupedGemm_VariableSizes(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)

        dist = torch.rand(z, )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)

        out = ops.gmm(a, b, batch_sizes, trans_b)
        expected_out = gmm(a, b, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))



if __name__ == '__main__':
    unittest.main()
