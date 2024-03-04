cmake_minimum_required(VERSION 3.1...3.27)

project(
  StbImageProj
  VERSION 1.0
  LANGUAGES CXX CUDA)

add_library(StbImage STATIC image.cpp)
target_include_directories(StbImage INTERFACE ./)