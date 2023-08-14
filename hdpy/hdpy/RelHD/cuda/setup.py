from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hd_enc_cuda',
    ext_modules=[
        CUDAExtension('hd_enc_cuda', [
            'hd_enc_cuda.cpp',
            'hd_enc_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })