#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#include "generic_cuda_types.h"

// Define a macro for generating an array index for accessing a 2D row-major matrix stored in a flat array.
// The syntax is MatId(x, y, row width), and the calculation is index = (y * row width) + x
#define MatId(a, b, c) (__mul24((b), (c)) + (a))

namespace CudaUtil
{
	// Fill a shared memory array with image data, padded with zeroes at the borders.
	// The dimensions of the array should be the number of threads per block plus 2 times the border.
	template <typename T>
	__device__ void FillSmemFromSurface(
		T* smemArray,
		CudaUtil::Dim2 smemDim,
		cudaSurfaceObject_t imageSurface,
		CudaUtil::Dim2 imgDim,
		CudaUtil::KernelIndex index,
		dim3 blockSize,
		CudaUtil::Dim2 border
		)
	{
		const uint32_t twoBorderX = (2 * border.x);
		const uint32_t twoBorderY = (2 * border.y);
		const uint32_t dimX = smemDim.x;

		// Fill the border of the shared memory with zeroes.
		// Index (0, 0) is assumed to be top left.
		
		// Left border
		if (index.tx_l < border.x) {
			smemArray[MatId(index.tx_l, (index.ty_l + border.y), dimX)] = 0;
		}
		// Right border
		else if (index.tx_l >= (blockSize.x - border.x)) {
			smemArray[MatId((index.tx_l + twoBorderX), (index.ty_l + border.y), dimX)] = 0;
		}

		// Top border
		if (index.ty_l < border.y) {
			smemArray[MatId((index.tx_l + border.x), index.ty_l, dimX)] = 0;

			// Top left corner
			if (index.tx_l < border.x) {
				smemArray[MatId(index.tx_l, index.ty_l, dimX)] = 0;
			}
			// Top right corner
			else if (index.tx_l >= (blockSize.x - border.x)) {
				smemArray[MatId((index.tx_l + twoBorderX), index.ty_l, dimX)] = 0;
			}
		}
		// Bottom border
		else if (index.ty_l >= (blockSize.y - border.y)) {
			smemArray[MatId((index.tx_l + border.x), (index.ty_l + twoBorderY), dimX)] = 0;

			// Bottom left corner
			if (index.tx_l < border.x)	{
				smemArray[MatId(index.tx_l, (index.ty_l + twoBorderY), dimX)] = 0;
			}
			// Bottom right corner
			else if (index.tx_l >= (blockSize.x - border.x)) {
				smemArray[MatId((index.tx_l + twoBorderX), (index.ty_l + twoBorderY), dimX)] = 0;
			}
		}

		// Fill shared memory with image data.

		// Center
		surf2Dread(&smemArray[MatId((index.tx_l + border.x), (index.ty_l + border.y), dimX)], imageSurface, index.tx_g * sizeof(T), index.ty_g);

		// Left border
		if ((index.tx_l < border.x) && (index.tx_g >= border.x)) {
			surf2Dread(&smemArray[MatId(index.tx_l, (index.ty_l + border.y), dimX)], imageSurface, (index.tx_g - border.x) * sizeof(T), index.ty_g);
		}
		// Right border
		else if ((index.tx_l >= (blockSize.x - border.x)) && (index.tx_g < (imgDim.x - border.x))) {
			surf2Dread(&smemArray[MatId((index.tx_l + twoBorderX), (index.ty_l + border.y), dimX)], imageSurface, (index.tx_g + border.x) * sizeof(T), index.ty_g);
		}
		
		// Top border
		if ((index.ty_l < border.y) && (index.ty_g >= border.x)) {
			surf2Dread(&smemArray[MatId((index.tx_l + border.x), index.ty_l, dimX)], imageSurface, index.tx_g * sizeof(T), (index.ty_g - border.x));

			// Top left corner
			if ((index.tx_l < border.x) && ((index.tx_g >= border.x))) {
				surf2Dread(&smemArray[MatId(index.tx_l, index.ty_l, dimX)], imageSurface, (index.tx_g - border.x) * sizeof(T), (index.ty_g - border.y));
			}
			// Top right corner
			else if ((index.tx_l >= blockSize.x - border.x) && (index.tx_g < (imgDim.x - border.x))) {
				surf2Dread(&smemArray[MatId((index.tx_l + twoBorderX), index.ty_l, dimX)], imageSurface, (index.tx_g + border.x) * sizeof(T), (index.ty_g - border.y));
			}
		}
		// Bottom border
		else if ((index.ty_l >= (blockSize.y - border.y)) && (index.ty_g < (imgDim.y - border.y))) {
			surf2Dread(&smemArray[MatId((index.tx_l + border.x), (index.ty_l + twoBorderY), dimX)], imageSurface, index.tx_g * sizeof(T), (index.ty_g + border.y));

			// Bottom left corner
			if ((index.tx_l < border.x) && ((index.tx_g >= border.x))) {
				surf2Dread(&smemArray[MatId(index.tx_l, (index.ty_l + twoBorderY), dimX)], imageSurface, (index.tx_g - border.x) * sizeof(T), (index.ty_g - border.y));
			}

			// Bottom right corner
			else if ((index.tx_l >= (blockSize.x - border.x)) && (index.tx_g < (imgDim.y - border.x))) {
				surf2Dread(&smemArray[MatId((index.tx_l + twoBorderX), (index.ty_l + twoBorderY), dimX)], imageSurface, (index.tx_g + border.x) * sizeof(T), (index.ty_g + border.y));
			}
		}
	}
}
