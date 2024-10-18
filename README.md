# Grouped GEMM

A lighweight library exposing grouped GEMM kernels in PyTorch.

# Installation

Run `pip install grouped_gemm` to install the package.

# Compiling from source

By default, the installed package runs in conservative (`cuBLAS`) mode:
it launches one GEMM kernel per batch element instead of using a single
grouped GEMM kernel for the whole batch.

To enable using grouped GEMM kernels, you need to switch to the `CUTLASS`
mode by setting the `GROUPED_GEMM_CUTLASS` environment variable to `1`
when building the library. For example, to build the library in `CUTLASS`
mode for Ampere (SM 8.0), clone the repository and run the following:

```bash
$ TORCH_CUDA_ARCH_LIST=8.0 GROUPED_GEMM_CUTLASS=1 pip install .
```

See [this comment](https://github.com/tgale96/grouped_gemm/pull/14#issuecomment-2211362572)
for some performance measurements on A100 and H100.

# Upcoming features

* Hopper-optimized grouped GEMM kernels.
