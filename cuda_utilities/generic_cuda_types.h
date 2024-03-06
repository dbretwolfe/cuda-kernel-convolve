#pragma once

#include <cstdint>

namespace CudaUtil
{
    // Config struct to hold image dimensions for passing to functions and kernels.
	typedef struct
	{
		uint32_t x;
		uint32_t y;
	} Dim2;

	// Config struct for passing thread indices to kernels and device functions
	typedef struct
	{
		uint32_t tx_l;
		uint32_t ty_l;
		uint32_t tx_g;		
		uint32_t ty_g;
	} KernelIndex;

    typedef struct
	{
		uint32_t x;
		uint32_t y;
        uint32_t z;
	} CudaBlockSize;
}