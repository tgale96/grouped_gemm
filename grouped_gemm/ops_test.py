import unittest
import itertools
import numpy as np
import torch
from absl.testing import parameterized

from grouped_gemm import ops


def assert_allclose(a, b, rtol=5e-03, atol=0):
    np.testing.assert_allclose(
        a.float().cpu().numpy(),
        b.float().cpu().numpy(),
        rtol=rtol, atol=atol,
    )


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
        assert_allclose(out, expected_out)

if __name__ == '__main__':
    unittest.main()
