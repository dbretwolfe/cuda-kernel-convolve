cmake_minimum_required(VERSION 3.1...3.27)

project(
  CudaConvolveProj
  VERSION 1.0
  LANGUAGES CXX CUDA)

add_library(CudaConvolve STATIC)
target_sources(CudaConvolve PUBLIC cuda_convolve.cpp)
target_include_directories(CudaConvolve PUBLIC ./ ../cuda_utilities/ ../convolve_kernel/)
target_link_libraries(CudaConvolve PUBLIC ConvolveKernel)
target_compile_features(CudaConvolve PUBLIC cxx_std_11)
set_target_properties(CudaConvolve PROPERTIES 
						CUDA_SEPARABLE_COMPILATION ON
						CMAKE_POSITION_INDEPENDENT_CODE ON
					)