cmake_minimum_required(VERSION 3.1...3.27)

project(
  ConvolveKernelProj
  VERSION 1.0
  LANGUAGES CXX CUDA)

add_library(ConvolveKernel STATIC)
target_sources(ConvolveKernel PUBLIC convolve_kernel.cu)
target_include_directories(ConvolveKernel PUBLIC ./ ../cuda_utilities/ ../fill_smem_from_surface/)
target_link_libraries(ConvolveKernel PUBLIC FillSmemFromSurface)
target_compile_features(ConvolveKernel PUBLIC cxx_std_11)
set_target_properties(ConvolveKernel PROPERTIES 
						CUDA_SEPARABLE_COMPILATION ON
						CMAKE_POSITION_INDEPENDENT_CODE ON
					)