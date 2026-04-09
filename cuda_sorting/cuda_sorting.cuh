#pragma once

#include <stdint.h>
#include <math.h>

#include "cuda_runtime.h"

namespace CudaSorting
{
    // Swap the values of float a and float b.
    template<typename T>
    void Swap(T* a, T* b)
    {
        T temp = *a;
        *a = *b;
        *b = temp;
    }

    // Quicksort-style partition.  The rightmost array element is selected as the pivot,
    // and all values less than the pivot are moved to the left.
    template<typename T>
    size_t Partition(T* values, size_t leftIndex, size_t rightIndex)
    {
        T pivot = values[rightIndex];
        size_t pivotIndex = leftIndex;

        for (size_t i = leftIndex; i <= rightIndex - 1; i++)
        {
            if (values[i] <= pivot)
            {
                Swap(&values[pivotIndex], &values[i]);
                pivotIndex++;
            }
        }

        Swap(&values[pivotIndex], &values[rightIndex]);

        return pivotIndex;
    }

    // Recursively determine the value of the k'th smallest element in the values list within the left
    // and right indices.
    template<typename T>
    T QuickSelect(T* values, size_t leftIndex, size_t rightIndex, size_t k)
    {
        // If the array contains only one element, return that element.
        if (leftIndex == rightIndex)
        {
            return values[leftIndex];
        }

        size_t pivotIndex = Partition(values, leftIndex, rightIndex);

        // If the pivot is in it's final sorted position, return the value of the pivot index.
        if (k == pivotIndex)
        {
            return values[k];
        }

        // If `k` is less than the pivot index, recurse, shrinking the list one element from the right.
        else if (k < pivotIndex)
        {
            return QuickSelect(values, leftIndex, pivotIndex - 1, k);
        }

        // If `k` is more than the pivot index,recurse, shrinking the list one element from the left.
        else
        {
            return QuickSelect(values, pivotIndex + 1, rightIndex, k);
        }
    }

    // Return the median value in an array of floats.
    template<typename T>
    T Median(T* values, size_t len)
    {
        // Set the rank for quickselect to the middle index value of the array.
        size_t k = (len / 2);

        // If the length of the values array is odd, the value at rank k is the median.
        // Otherwise, average the two middle values.
        if ((len % 2) == 1)
        {
            return QuickSelect(values, 0, (len - 1), k);
        }
        else
        {
            QuickSelect(values, 0, (len - 1), k);
            return ((values[k] + values[k - 1]) / 2);
        }
    }

    // Swap the values of float a and float b.
    template<typename T>
    __device__ void CudaSwap(T* a, T* b)
    {
        T temp = *a;
        *a = *b;
        *b = temp;
    }

    // Quicksort-style partition.  The rightmost array element is selected as the pivot,
    // and all values less than the pivot are moved to the left.
    template<typename T>
    __device__ size_t CudaPartition(T* values, size_t leftIndex, size_t rightIndex)
    {
        T pivot = values[rightIndex];
        size_t pivotIndex = leftIndex;

        for (size_t i = leftIndex; i <= rightIndex - 1; i++)
        {
            if (values[i] <= pivot)
            {
                CudaSwap(&values[pivotIndex], &values[i]);
                pivotIndex++;
            }
        }

        CudaSwap(&values[pivotIndex], &values[rightIndex]);

        return pivotIndex;
    }

    // Recursively determine the value of the k'th smallest element in the values list within the left
    // and right indices.
    template<typename T>
    __device__ T CudaQuickSelect(T* values, size_t leftIndex, size_t rightIndex, size_t k)
    {
        // If the array contains only one element, return that element.
        if (leftIndex == rightIndex)
        {
            return values[leftIndex];
        }

        size_t pivotIndex = CudaPartition(values, leftIndex, rightIndex);

        // If the pivot is in it's final sorted position, return the value of the pivot index.
        if (k == pivotIndex)
        {
            return values[k];
        }

        // If `k` is less than the pivot index, recurse, shrinking the list one element from the right.
        else if (k < pivotIndex)
        {
            return CudaQuickSelect(values, leftIndex, pivotIndex - 1, k);
        }

        // If `k` is more than the pivot index,recurse, shrinking the list one element from the left.
        else
        {
            return CudaQuickSelect(values, pivotIndex + 1, rightIndex, k);
        }
    }

    // Bubble sort algorithm
    template<typename T>
    __device__ void CudaBubbleSort(T* values, size_t len)
    {
        for (size_t i = 0; i < len - 1; i++)
        {
            // Last i elements are already in place.
            for (size_t j = 0; j < len - i - 1; j++)
            {
                if (values[j] > values[j + 1])
                {
                    CudaSwap(&values[j], &values[j + 1]);
                }
            }
        }
    }

    // Return the median value in an array of floats.
    template<typename T>
    __device__ T CudaMedian(T* values, size_t len)
    {
        // Calculate the middle index of the values array.
        size_t middle = (len / 2);

        CudaBubbleSort(values, len);

        // If the length of the values array is odd, the value at the middle is the median.
        // Otherwise, average the two middle values.
        if ((len % 2) == 1)
        {
            return values[middle];
        }
        else
        {
            return ((values[middle] + values[middle - 1]) / 2);
        }
    }
};