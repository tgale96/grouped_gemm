import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if not torch.cuda.is_available():
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
_dc = torch.cuda.get_device_capability()
_dc = f"{_dc[0]}{_dc[1]}"

ext_modules = [
    CUDAExtension(
        "grouped_gemm_backend",
        ["csrc/ops.cu", "csrc/grouped_gemm.cu"],
        include_dirs = [
            f"{cwd}/third_party/cutlass/include/",
            f"{cwd}/csrc"
        ],
        extra_compile_args={
            "cxx": [
                "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
            ],
            "nvcc": [
                f"--generate-code=arch=compute_{_dc},code=sm_{_dc}",
                f"-DGROUPED_GEMM_DEVICE_CAPABILITY={_dc}",
                # NOTE: CUTLASS requires c++17.
                "-std=c++17",
            ],
        }
    )
]

extra_deps = {}

extra_deps['dev'] = [
    'absl-py',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(
    name="grouped_gemm",
    version="0.0.1",
    author="Trevor Gale",
    author_email="tgale@stanford.edu",
    description="Grouped GEMM",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/tgale06/grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    extras_require=extra_deps,
)
