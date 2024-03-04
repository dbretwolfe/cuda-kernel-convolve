#include <cstdint>
#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include "cuda_convolver.cuh"
#include "generic_cuda_types.h"
#include "image.h"

int main()
{
    std::vector<float> gaussian3x3Kernel = {
        1/16, 2/16, 1/16,
        2/16, 4/16, 2/16, 
        1/16, 2/16, 1/16
    };
    CudaUtil::Dim2 kernelDim = { 3, 3 };

    std::string filePath = "../../../images/finn.png";
    auto image = std::make_shared<StbImage::Image>(filePath);

    CudaImgProc::CudaConvolver convolver(image);
    //std::vector<uint32_t> outputImg = convolver.Convolve()

    return 0;
}