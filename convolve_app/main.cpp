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
        (1.0 / 16.0), (2.0 / 16.0), (1.0 / 16.0),
        (2.0 / 16.0), (4.0 / 16.0), (2.0 / 16.0),
        (1.0 / 16.0), (2.0 / 16.0), (1.0 / 16.0)
    };

    std::vector<float> gaussian5x5Kernel = {
        (1.0 / 256.0), (4.0 / 256.0), (6.0 / 256.0), (4.0 / 256.0), (1.0 / 256.0),
        (4.0 / 256.0), (16.0 / 256.0), (24.0 / 256.0), (16.0 / 256.0), (4.0 / 256.0),
        (6.0 / 256.0), (24.0 / 256.0), (36.0 / 256.0), (24.0 / 256.0), (6.0 / 256.0),
        (4.0 / 256.0), (16.0 / 256.0), (24.0 / 256.0), (16.0 / 256.0), (4.0 / 256.0),
        (1.0 / 256.0), (4.0 / 256.0), (6.0 / 256.0), (4.0 / 256.0), (1.0 / 256.0)
    };

    std::vector<float> edgeDetect3x3Kernel = {
        -1.0, -1.0, -1.0,
        -1.0, 8.0, -1.0,
        -1.0, -1.0, -1.0
    };   

    //CudaUtil::Dim2 kernelDim = { 3, 3 };
    CudaUtil::Dim2 kernelDim = { 5, 5 };

    std::string filePath = "../../../images/finn.png";
    auto inputImage = std::make_shared<StbImage::Image>(filePath);

    CudaImgProc::CudaConvolver convolver(inputImage);
    auto outputImg = convolver.Convolve(gaussian5x5Kernel, kernelDim);
    outputImg.Write("../../../images/output.png", StbImage::Image::ImageType::PNG, 100);

    return 0;
}