#include <stdexcept>

#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace StbImage
{
    Image::Image(std::string filePath)
    {
        data = stbi_load(filePath.c_str(), &_width, &_height, &_channels, 0);
        if (!data) {
            throw std::runtime_error(stbi_failure_reason());
        }
    }

    Image::~Image()
    {
        if (data) {
            stbi_image_free(data);
        }
    }
}