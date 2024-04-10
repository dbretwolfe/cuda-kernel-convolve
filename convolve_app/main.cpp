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
        1.0/16.0, 2.0/16.0, 1.0/16.0,
        2.0/16.0, 4.0/16.0, 2.0/16.0, 
        1.0/16.0, 2.0/16.0, 1.0/16.0
    };
    CudaUtil::Dim2 kernelDim = { 3, 3 };

    std::string filePath = "../../../images/finn.png";
    auto inputImage = std::make_shared<StbImage::Image>(filePath);

    CudaImgProc::CudaConvolver convolver(inputImage);
    auto outputImg = convolver.Convolve(gaussian3x3Kernel, kernelDim);
    outputImg.Write("../../../images/output.png", StbImage::Image::ImageType::PNG, 100);

    return 0;
}