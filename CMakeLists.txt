######## 基础声明
cmake_minimum_required(VERSION 3.16)
project(DigitalHumanSDK VERSION 0.1.0 LANGUAGES CXX)

######## 全局配置与编译优化
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认开启优化
add_compile_options(-O3)

# 启用本机指令集优化
add_compile_options(-march=native)
add_definitions(-DNDEBUG)

# 常用警告
add_compile_options(-Wall -Wextra)

######## 寻找并配置 OpenMP 多线程加速（视系统环境而定）
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

######## 第三方查找库
# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 打印以下信息，方便调试
message(STAUS "OpenCV library status:")
message(STAUS "OpenCV version: ${OpenCV_VERSION}")
message(STAUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# 将 OpenCV 的头文件路径添加到全局包含路径中
include_directories(${OpenCV_INCLUDE_DIRS}) 
include_directories(${CMKAE_SOURCE_DIR}/include)

# 查找 ncnn 库
find_package(ncnn REQUIRED)
message(STATUS ">> ncnn Found")

# 查找 FFmpeg
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

# 将 FFmpeg 的头文件路径添加到全局包含路径中
include_directories(${FFmpeg_INCLUDE_DIRS})

# 查找 dlib
find_package(dlib REQUIRED)
message(STATUS ">> dlib Found: ${dlib_VERSION}")
# dlib::dlib target 会自动处理 include path
include_directories(${dlib_INCLUDE_DIRS})

# 查找 spdlog 日志库
find_package(spdlog REQUIRED)
message(STATUS ">> spdlog Found: ${spdlog_VERSION}")

# 引入 pkg-config 工具
find_package(PkgConfig REQUIRED)
# 查找 portaudio-2.0
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
# 告诉编译器去哪里找 <portaudio.h>
include_directories(${PORTAUDIO_INCLUDE_DIRS})


######## 规范输出目录：编译出的库和可执行文件分别归类到 build/lib 和 build/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

######## 全局头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

######## 第三方依赖

# 项目的第三方依赖


######## 添加子目录分发任务
add_subdirectory(src)
add_subdirectory(examples)


