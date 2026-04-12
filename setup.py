from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tensorrt_op',
    ext_modules=[
        CUDAExtension(
            name='tensorrt_op',
            sources=['tensorrt_op.cpp'],
            libraries=['nvinfer', 'cudart'],
            library_dirs=['C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.x/lib'],
            include_dirs=['C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.x/include']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)