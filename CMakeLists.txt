# Builds the cuPQC answer inside the challenge image (fherma/cupqc), which
# provides CUDA 12.8 and the cuPQC SDK. Unlike the default generated C++
# CMakeLists, this one enables the CUDA language and links cuPQC BigInt.
cmake_minimum_required(VERSION 3.24)
project(fherma_solution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Release unless told otherwise: an answer measured at -O0 is not a measurement
# of the answer.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The GPU architecture to build for: the card the build can see. The platform
# builds a solution on the machine that measures it, with the GPU visible, so
# `native` compiles for an L40S (sm_89) there and for an RTX PRO 6000
# (sm_120) here — where a fixed sm_89 with device LTO shipped no PTX and had
# no kernel to run. Override with -DFHERMA_CUDA_ARCH=sm_89 to build for a card
# that is not present. CMake's own CMAKE_CUDA_ARCHITECTURES is left OFF because
# it emits -gencode, which nvcc refuses alongside -dlto; the architecture is
# passed as -arch= instead.
set(FHERMA_CUDA_ARCH native CACHE STRING "nvcc -arch value")

# CUPQC_SDK_DIR is set by the image; fall back to the default install location.
if(DEFINED ENV{CUPQC_SDK_DIR})
  set(CUPQC_SDK_DIR $ENV{CUPQC_SDK_DIR})
else()
  set(CUPQC_SDK_DIR /usr/local/cupqc-sdk)
endif()

add_executable(solution main.cpp solve.cu)

target_include_directories(solution PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUPQC_SDK_DIR}/include
  ${CUPQC_SDK_DIR}/include/cupqc)

# The cuPQC BigInt implementation ships as device-LTO objects, so the device
# code is compiled and device-linked with -dlto, and the static library is put
# on the device link line. -arch=<sm> replaces CMAKE_CUDA_ARCHITECTURES, whose
# -gencode form conflicts with -dlto.
set_target_properties(solution PROPERTIES
  CUDA_ARCHITECTURES OFF
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(solution PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${FHERMA_CUDA_ARCH};-dlto>)
target_link_options(solution PRIVATE
  $<DEVICE_LINK:-arch=${FHERMA_CUDA_ARCH};-dlto>)

target_link_libraries(solution PRIVATE ${CUPQC_SDK_DIR}/lib/libcupqc-bigint.a)
