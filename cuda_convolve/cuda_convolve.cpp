#include <exception>
#include "cuda_convolve.h"

namespace CudaConvolve
{
    CudaConvolve::CudaConvolve(CudaUtil::ImgDim imgDim) : ._imgDim(imgDim)
    {
        cudaError cudaStatus = AllocCudaMem();
        if (cudaStatus != cudaSuccess) {
            throw new std::bad_alloc("Could not allocate memory for CUDA arrays!");
        }
    }

    CudaConvolve::~CudaConvolve()
    {

    }

    cudaError CudaConvolve::AllocCudaMem()
    {
        cudaError cudaStatus;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

        // Allocate the cuda arrays for input and output.
		cudaStatus = cudaMallocArray(&_devInputArray,
			&channelDesc,
			_imgDim.x,
			_imgDim.y,
			cudaArraySurfaceLoadStore);
		if (cudaStatus != cudaSuccess) { return cudaStatus; }
    }
}