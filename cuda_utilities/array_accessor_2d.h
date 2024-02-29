#pragma once

#include <cstdint>

namespace CudaUtil 
{
    template <typename T, size_t arraySize>
    class ArrayAccessor2d
    {
    public:
        ArrayAccessor2d(uint32_t dimX, uint32_t dimY, T (&array)[arraySize] :
        dimX(dimX),
        dimY(dimY),
        _array(array) {}

        T& operator()(uint32_t x, uint32_t y)
        {
            return _array[(y * dimX) + x];
        }

    private:
        uint32_t dimX;
        uint32_t dimY;
        T (&_array)[arraySize];
    };
}