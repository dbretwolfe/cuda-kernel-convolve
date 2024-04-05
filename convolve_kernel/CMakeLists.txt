add_library(ConvolveKernel STATIC convolve_kernel.cu)
target_include_directories(ConvolveKernel PUBLIC ./ ../cuda_utilities/ ../fill_smem_from_surface/)
target_link_libraries(ConvolveKernel PUBLIC FillSmemFromSurface)
target_compile_features(ConvolveKernel PUBLIC cxx_std_11)