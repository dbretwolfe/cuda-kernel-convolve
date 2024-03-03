cmake_minimum_required(VERSION 3.1...3.27)

project(
  CudaConvolveProj
  VERSION 1.0
  LANGUAGES CXX CUDA)

add_library(CudaConvolver STATIC cuda_convolver.cu)
#target_sources(CudaConvolver PUBLIC cuda_convolver.cu)
target_include_directories(CudaConvolver PUBLIC ./ ../cuda_utilities/ ../convolve_kernel/)
target_link_libraries(CudaConvolver PUBLIC ConvolveKernel)
target_compile_features(CudaConvolver PUBLIC cxx_std_11)