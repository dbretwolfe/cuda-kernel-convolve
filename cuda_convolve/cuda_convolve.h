#pragma once

#include <vector>

#include "convolve_kernel.cuh"

namespace CudaConvolve
{
    class CudaConvolve
    {
    public:
        CudaConvolve(CudaUtil::ImgDim imgDim);
        ~CudaConvolve();
        void Convolve();

    private:
        cudaError AllocCudaMem();

        const ImgDim _imgDim;
        cudaArray_t _devInputArray;
        cudaArray_t _devOutputArray;
        cudaSurfaceObject_t _devInputSurface;
        cudaSurfaceObject_t _devOutputSurface;
    };
}