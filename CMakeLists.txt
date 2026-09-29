cmake_minimum_required(VERSION 3.16)
project(DigitalHumanSDK VERSION 0.1.0 LANGUAGES CXX)

######################### 全局配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认开启优化。固定为通用 x86-64，避免构建机本机指令集
# （如 AVX2/AVX512）被带入需要跨虚拟 CPU 运行的二进制。
add_compile_options(-O3 -march=x86-64 -mtune=generic)
add_definitions(-DNDEBUG)

# 常用警告
add_compile_options(-Wall -Wextra)

######################### OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

######################### 输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

######################### 全局头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

######################### 第三方依赖

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# ncnn
find_package(ncnn REQUIRED)
message(STATUS ">> ncnn Found")

# FFmpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswresample
    libswscale
)

message(STATUS ">> FFMPEG Found: ${FFMPEG_VERSION}")
message(STATUS ">> FFmpeg Include: ${FFMPEG_INCLUDE_DIRS}")
include_directories(${FFMPEG_INCLUDE_DIRS})

# PortAudio
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
message(STATUS ">> PortAudio Found: ${PORTAUDIO_FOUND}")
include_directories(${PORTAUDIO_INCLUDE_DIRS})

# dlib
find_package(dlib REQUIRED)
message(STATUS ">> dlib Found: ${dlib_VERSION}")
include_directories(${dlib_INCLUDE_DIRS})

######################### 子目录
add_subdirectory(src)
add_subdirectory(examples)

######################### 测试
# add_subdirectory(test)
