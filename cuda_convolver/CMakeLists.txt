add_library(CudaConvolver STATIC cuda_convolver.cu)
target_include_directories(CudaConvolver PUBLIC ./ ../cuda_utilities/ ../convolve_kernel/ ../stb/)
target_link_libraries(CudaConvolver PUBLIC ConvolveKernel StbImage)
target_compile_features(CudaConvolver PUBLIC cxx_std_11)