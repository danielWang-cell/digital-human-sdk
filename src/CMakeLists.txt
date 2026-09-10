# 自动搜索源文件
# 使用 GLOB_RECURSE 递归查找 src 下的所有子目录的 .cpp 文件
file(GLOB_RECURSE SRC_FILES
    "audio/*.cpp"
    "video/*.cpp"
    "core/*.cpp"
    "utils/*.cpp"
    "model/*.cpp"
)

# 生成 SDK 库
# 将所有源文件编译成一个名为 digital_human_core 的动态链接库（SHARED）
add_library(digital_human_core SHARED ${SRC_FILES})

# 配置库的头文件搜索路径，并声明为 PUBLIC（让链接它的上层应用也能自动共享该路径）
target_include_directories(digital_human_core PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
)

# 配置一些第三方库

# 链接 OpenCV 库
target_link_libraries(digital_human_core PUBLIC
    ${OpenCV_LIBS}
)

message("OpenCV libraries: ${OpenCV_LIBS}")

# 链接 OpenCV ncnn FFMPEG dlib 库
target_link_libraries(digital_human_core PUBLIC
    ${OpenCV_LIBS}
    ncnn
    PkgConfig::FFMPEG
    ${PORTAUDIO_LIBRARIES}
    dlib::dlib
    spdlog::spdlog
)


