cmake_minimum_required(VERSION 3.1...3.27)

project(
  ConvolveAppProj
  VERSION 1.0
  LANGUAGES CXX CUDA)

add_executable(ConvolveApp)
target_sources(ConvolveApp PUBLIC main.cpp)
target_include_directories(ConvolveApp PUBLIC ./ ../cuda_utilities/ ../cuda_convolve/)
target_link_libraries(ConvolveApp PUBLIC CudaConvolve)
target_compile_features(ConvolveApp PUBLIC cxx_std_11)
set_target_properties(ConvolveApp PROPERTIES 
						CUDA_SEPARABLE_COMPILATION ON
						CMAKE_POSITION_INDEPENDENT_CODE ON
					)