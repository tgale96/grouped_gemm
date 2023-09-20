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


_TEST_PROBLEMS = (
    (1, 128, 128, 128),
    (8, 128, 128, 128),
    (16, 128, 128, 128),
    (1, 128, 256, 512),
    (8, 128, 256, 512),
    (16, 128, 256, 512),
)


def randn(bs, x, y):
    out = (torch.rand(bs, x, y) - 0.5 * 2) / (y * x)
    return out.cuda().to(torch.bfloat16)


@parameterized.parameters(*_TEST_PROBLEMS)
class OpsTest(parameterized.TestCase):

    def testGroupedGemm(self, z, m, k, n):
        torch.manual_seed(0)
        a = randn(z, m, k)
        b = randn(z, k, n)
        batch_sizes = torch.tensor([m] * z)

        out = ops.grouped_gemm(a.view(-1, k), b, batch_sizes).view(z, m, n)
        expected_out = torch.bmm(a, b)
        self.assertTrue(allclose(out, expected_out))


if __name__ == '__main__':
    unittest.main()
