#pragma once

#include <string>
#include <cstdint>

namespace StbImage
{
    class Image
    {
    public:
        enum class ImageType {PNG, JPG};

        Image(uint32_t width, uint32_t height, uint32_t channels);
        Image(std::string filePath);
        Image(const Image& image);
        ~Image();

        void Write(const std::string filePath, ImageType imageType, uint8_t quality = 100);

        uint32_t width() const { return static_cast<uint32_t>(_width); }
        uint32_t height() const { return static_cast<uint32_t>(_height); }
        uint32_t channels() const { return static_cast<uint32_t>(_channels); }
        uint32_t size() const { return _size; }
        
        uint8_t* data = nullptr;

    private:
        int _width = 0;
        int _height = 0;
        int _channels = 0;
        size_t _size = 0;
    };
}