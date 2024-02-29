cmake_minimum_required(VERSION 3.1...3.27)

project(
  FillSmemFromSurfaceProj
  VERSION 1.0
  LANGUAGES CXX CUDA)

add_library(FillSmemFromSurface STATIC)
target_sources(FillSmemFromSurface PUBLIC fill_smem_from_surface.cu)
target_include_directories(FillSmemFromSurface PUBLIC ./ ../cuda_utilities/)
target_compile_features(FillSmemFromSurface PUBLIC cxx_std_11)
set_target_properties(FillSmemFromSurface PROPERTIES 
						CUDA_SEPARABLE_COMPILATION ON
						CMAKE_POSITION_INDEPENDENT_CODE ON
					)