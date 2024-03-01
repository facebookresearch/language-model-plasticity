# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="dynamicconv_layer",
    ext_modules=[
        CUDAExtension(
            name="dynamicconv_cuda",
            sources=[
                "dynamicconv_cuda.cpp",
                "dynamicconv_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
