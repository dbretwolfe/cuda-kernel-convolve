#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "generic_cuda_types.h"

namespace ConvolveKernels
{
    __global__ void ConvolveRgba(
        cudaSurfaceObject_t inputSurface,
        cudaSurfaceObject_t outputSurface,
        CudaUtil::Dim2 imgDim,
        float* kernel,
        CudaUtil::Dim2 kernelDim,
        CudaUtil::Dim2 smemDim
    );
}