add_library(FillSmemFromSurface INTERFACE)
target_include_directories(FillSmemFromSurface INTERFACE ./ ../cuda_utilities/ ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})