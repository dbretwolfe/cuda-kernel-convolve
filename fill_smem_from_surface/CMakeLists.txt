cmake_minimum_required(VERSION 3.1...3.27)

project(
  FillSmemFromSurfaceProj
  VERSION 1.0
  LANGUAGES CXX CUDA)

add_library(FillSmemFromSurface INTERFACE)
target_include_directories(FillSmemFromSurface INTERFACE ./ ../cuda_utilities/)