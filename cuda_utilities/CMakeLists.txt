cmake_minimum_required(VERSION 3.1...3.27)

project(
  CudaUtilsProj
  VERSION 1.0
  LANGUAGES CXX CUDA)

add_library(CudaUtils INTERFACE)
target_include_directories(CudaUtils INTERFACE ./)