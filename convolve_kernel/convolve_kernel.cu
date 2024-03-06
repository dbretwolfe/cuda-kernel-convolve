#include "convolve_kernel.cuh"
#include "fill_smem_from_surface.cuh"

#define IMAD(a, b, c) (__mul24((a), (b)) + (c))

namespace ConvolveKernels
{
    typedef union 
    {
        uint32_t u32Data;
        uint8_t channels[4];
    } RgbaData;
    
    __global__ void ConvolveRgba(
        cudaSurfaceObject_t inputSurface,
        cudaSurfaceObject_t outputSurface,
        CudaUtil::Dim2 imgDim,
        float* kernel,
        CudaUtil::Dim2 kernelDim,
        CudaUtil::Dim2 smemDim
    )
    {
        // Local thread indices
        const uint32_t tx_l = threadIdx.x;
        const uint32_t ty_l = threadIdx.y;

        // Global thread indices
        const uint32_t tx_g = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
        const uint32_t ty_g = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

        extern __shared__ uint32_t smemArray[];

        CudaUtil::KernelIndex index = { tx_l, ty_l, tx_g, ty_g };

        // The border added to the shared memory array should be equal to the kernel radius.
        CudaUtil::Dim2 kernelRadius = { (kernelDim.x / 2), (kernelDim.y / 2) };

        CudaUtil::FillSmemFromSurface<uint32_t>(
            smemArray,
            inputSurface,
            imgDim,
            index,
            blockDim,
            kernelRadius
        );

        __syncthreads();

        // Apply the kernel to the image data.
        // Column first traversal results in better cacheing.
        float rAccum = 0;
        float gAccum = 0;
        float bAccum = 0;
        float aAccum = 0;

        for (int y = 0; y <= kernelDim.y; y++) {
            for (int x = 0; y <= kernelDim.x; x++) {
                RgbaData inputPixel;
                inputPixel.u32Data = smemArray[MatId((tx_l + x), (ty_l + y), smemDim.x)];
                rAccum += static_cast<float>(inputPixel.channels[0]) * kernel[MatId(x, y, kernelDim.x)];
                gAccum += static_cast<float>(inputPixel.channels[1]) * kernel[MatId(x, y, kernelDim.x)];
                bAccum += static_cast<float>(inputPixel.channels[2]) * kernel[MatId(x, y, kernelDim.x)];
                aAccum += static_cast<float>(inputPixel.channels[3]) * kernel[MatId(x, y, kernelDim.x)];
            }
        }

        RgbaData outputPixel;
        outputPixel.channels[0] = static_cast<uint8_t>(min(255.0, rAccum));
        outputPixel.channels[1] = static_cast<uint8_t>(min(255.0, rAccum));
        outputPixel.channels[2] = static_cast<uint8_t>(min(255.0, rAccum));
        outputPixel.channels[3] = static_cast<uint8_t>(min(255.0, rAccum));

        surf2Dwrite(outputPixel.u32Data, outputSurface, tx_g * sizeof(uint32_t), ty_g);
    }
}