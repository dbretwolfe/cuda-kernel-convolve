#include <stdexcept>

#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace StbImage
{
    // Create empty image
    Image::Image(uint32_t width, uint32_t height, uint32_t channels)
    {
        if ((width == 0) || (height == 0) || (channels == 0)) {
            throw std::invalid_argument("Invalid image parameters!");
        }

        data = new uint8_t[width * height * channels];

        _width = width;
        _height = height;
        _channels = channels;
        _size = _width * _height * _channels;
    }

    // Load image from file
    Image::Image(std::string filePath)
    {
        data = stbi_load(filePath.c_str(), &_width, &_height, &_channels, 0);
        if (data == nullptr) {
            throw std::runtime_error(stbi_failure_reason());
        }

        _size = _width * _height * _channels;
    }

    // Copy existing image object, delegating to the create empty image constructor for allocation.
    Image::Image(const Image& image) : Image(image.width(), image.height(), image.channels())
    {
        std::copy(image.data, (image.data + _size), this->data);
    }

    Image::~Image()
    {
        if (data) {
            stbi_image_free(data);
        }
    }

    void Image::Write(const std::string filePath, ImageType imageType, uint8_t quality)
    {
        if (!data) {
            throw std::runtime_error("Image data is invalid, cannot write!");
        }
        if (filePath.empty()) {
            throw std::invalid_argument("Invalid file path, cannot write!");
        }

        switch (imageType) {
            case ImageType::PNG:
                stbi_write_png(filePath.c_str(), _width, _height, _channels, data, (_width * _channels));
                break;

            case ImageType::JPG:
                stbi_write_jpg(filePath.c_str(), _width, _height, _channels, data, quality);
        }
    }
}