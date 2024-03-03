#include <cstdint>
#include <iostream>
#include <vector>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "cuda_convolver.cuh"

int main()
{
    std::vector<float> gaussian3x3Kernel = {
        1/16, 2/16, 1/16,
        2/16, 4/16, 2/16, 
        1/16, 2/16, 1/16
    };
    CudaUtil::Dim2 kernelDim = { 3, 3 };

    int width = 0;
    int height = 0;
    int channels = 0;
    const char* filename = "../../../images/finn.png";
    auto img = std::unique_ptr<uint8_t>(stbi_load(filename, &width, &height, &channels, 0));
    if (!img) {
        std::cout << stbi_failure_reason() << std::endl;
    }

    CudaUtil::Dim2 imgDim = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    size_t imgSize = width * height * 4;
    CudaImgProc::CudaConvolver convolver(imgDim);
    //std::vector<uint32_t> outputImg = convolver.Convolve()

    return 0;
}