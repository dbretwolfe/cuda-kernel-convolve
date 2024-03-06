#include <new>
#include <iostream>

#include "cuda_convolver.cuh"
#include "convolve_kernel.cuh"

#include "cuda_check.h"

namespace CudaImgProc
{
    CudaConvolver::CudaConvolver(std::shared_ptr<StbImage::Image> image) : _image(image)
    {
        cudaError cudaStatus = AllocCudaMem();
        if (cudaStatus != cudaSuccess) {
            throw std::exception("Could not allocate memory for CUDA arrays!");
        }
    }

    CudaConvolver::~CudaConvolver()
    {
        DeleteCudaMem();
    }

    StbImage::Image CudaConvolver::Convolve(std::vector<float>& kernel, CudaUtil::Dim2 kernelDim)
    {
        // Calculate the number of CUDA blocks needed.
		const dim3 numBlocks(
            IntDivUp(_image->width(), blockSize.x), 
            IntDivUp(_image->height(), blockSize.y)
            );

        const size_t arrayPitch = _image->width() * _image->channels();

        // Copy the input image data to the device.
        CUDA_CHECK(cudaMemcpy2DToArray(
            _devInputArray, 
            0, 
            0, 
            _image->data,
            arrayPitch,
            arrayPitch,
            _image->height(), 
            cudaMemcpyHostToDevice
            ));

        // Call the convolve CUDA kernel.
        CudaUtil::Dim2 imgDim = { _image->width(), _image->height() };
        CudaUtil::Dim2 kernelRadius = { (kernelDim.x / 2), (kernelDim.y / 2) };
        CudaUtil::Dim2 smemDim = { (blockSize.x + (2 * kernelRadius.x)), (blockSize.y + (2 * kernelRadius.y)) };
        size_t smemArraySize = smemDim.x * smemDim.y * sizeof(uint32_t);
        
        ConvolveKernels::ConvolveRgba<<<numBlocks, blockSize, smemArraySize>>>(
            _devInputSurface,
            _devOutputSurface,
            imgDim,
            kernel.data(),
            kernelDim,
            smemDim
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // Create a blank image with the same parameters as the source image.
        StbImage::Image output(_image->width(), _image->height(), _image->channels());

        // Copy the image data from the GPU to the output image.
        CUDA_CHECK(cudaMemcpy2DFromArray(
            output.data,
            arrayPitch,
            _devOutputArray,
            0,
            0,
            arrayPitch,
            _image->height(),
            cudaMemcpyDeviceToHost
            ));

        return output;
    }

    cudaError CudaConvolver::AllocCudaMem()
    {
        cudaError cudaStatus;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

        // Allocate the CUDA arrays for input and output.
		cudaStatus = cudaMallocArray(&_devInputArray,
			&channelDesc,
			_image->width(),
			_image->height(),
			cudaArraySurfaceLoadStore);
		if (cudaStatus != cudaSuccess) { return cudaStatus; }

        cudaStatus = cudaMallocArray(&_devOutputArray,
			&channelDesc,
			_image->width(),
			_image->height(),
			cudaArraySurfaceLoadStore);
		if (cudaStatus != cudaSuccess) { return cudaStatus; }

        // Create the CUDA surfaces.
        struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;

		resDesc.res.array.array = _devInputArray;
		cudaStatus = cudaCreateSurfaceObject(&_devInputSurface, &resDesc);
		if (cudaStatus != cudaSuccess) { return cudaStatus; }

        resDesc.res.array.array = _devOutputArray;
		cudaStatus = cudaCreateSurfaceObject(&_devOutputSurface, &resDesc);
		if (cudaStatus != cudaSuccess) { return cudaStatus; }

        return cudaSuccess;
    }

    void CudaConvolver::DeleteCudaMem()
    {
        cudaFreeArray(_devInputArray);
        cudaFreeArray(_devOutputArray);
        cudaDestroySurfaceObject(_devInputSurface);
        cudaDestroySurfaceObject(_devOutputSurface);
    }
}