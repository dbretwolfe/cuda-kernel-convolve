#pragma once

#include <vector>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "generic_cuda_types.h"
#include "image.h"

namespace CudaImgProc
{
    class CudaConvolver
    {
    public:
        CudaConvolver(std::shared_ptr<StbImage::Image> image);
        ~CudaConvolver();
        
        StbImage::Image CudaConvolver::Convolve(std::vector<float>& kernel, CudaUtil::Dim2 kernelDim);

    private:
        cudaError AllocCudaMem();
        void DeleteCudaMem();
        inline int IntDivUp(int a, int b) const { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

        std::shared_ptr<StbImage::Image> _image;
        cudaArray_t _devInputArray;
        cudaArray_t _devOutputArray;
        cudaSurfaceObject_t _devInputSurface;
        cudaSurfaceObject_t _devOutputSurface;
        const dim3 blockSize = { 16, 16 };
    };
}