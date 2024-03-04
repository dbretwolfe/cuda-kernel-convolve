#pragma once

#include <string>
#include <cstdint>

namespace StbImage
{
    class Image
    {
    public:
        Image(std::string filePath);
        // Do not allow copy constructor, it will cause the image to be freed when the temp goes out of scope.
        Image(const Image&) = delete;
        ~Image();

        uint32_t width() { return static_cast<uint32_t>(_width); }
        uint32_t height() { return static_cast<uint32_t>(_height); }
        uint32_t channels() { return static_cast<uint32_t>(_channels); }
        
        uint8_t* data = NULL;

    private:
        int _width = 0;
        int _height = 0;
        int _channels = 0;
    };
}