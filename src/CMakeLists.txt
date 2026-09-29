# 这个 CMakeLists.txt 文件用于处理源码
# 将 audio / video / core / model / utils 目录下的源码编译成一个 SDK 库，链接给上层使用

# 自动搜集源文件
file(GLOB_RECURSE SRC_FILES
    "audio/*.cpp"
    "video/*.cpp"
    "core/*.cpp"
    "utils/*.cpp"
    "model/*.cpp"
)

# 生成 SDK 动态库
add_library(digital_human_core SHARED ${SRC_FILES})

# 配置库的头文件搜索路径
target_include_directories(digital_human_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 链接第三方库
target_link_libraries(digital_human_core PUBLIC
    ${OpenCV_LIBS}
    ncnn
    PkgConfig::FFMPEG
    ${PORTAUDIO_LIBRARIES}
    dlib::dlib
)

# 如果检测到 OpenMP，则链接 OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(digital_human_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# 主程序
add_executable(digital_human_app main.cpp)

# 将主程序链接到 SDK 库
target_link_libraries(digital_human_app PRIVATE
    digital_human_core
)

# 实时数字人的目标
add_executable(digital_human_realtime_pulse_app main_realtime_pulse.cpp)

target_link_libraries(digital_human_realtime_pulse_app PRIVATE
    digital_human_core    
)