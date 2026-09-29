# ===============================================
# Digital Human SDK 终极开发环境 - Ubuntu 22.04
# 集成: OpenCV 4.5 + FFmpeg + ncnn + dlib + SSH
# ===============================================

FROM ubuntu:22.04

# 避免安装过程中的交互式弹窗
ENV DEBIAN_FRONTEND=noninteractive

# 1. 基础环境配置 (使用阿里云源加速)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 2. 安装所有系统级依赖
# build-essential: 基础编译工具
# libopencv-dev/ffmpeg: 图像视频处理
# libprotobuf-dev: ncnn 模型转换工具依赖
# libdlib-dev: dlib 库 (Ubuntu 22.04 自带 19.22)
# libsqlite3-dev: dlib 的链接依赖
# python3-pip: Python 依赖管理
RUN apt-get update && apt-get install -y \
    build-essential cmake gdb git wget curl vim nano zip unzip tar pkg-config ninja-build \
    openssh-server rsync locales ca-certificates \
    libopencv-dev ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev \
    libprotobuf-dev protobuf-compiler libvulkan-dev \
    libdlib-dev libopenblas-dev liblapack-dev libsqlite3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. 编译安装 ncnn (焊死在镜像里)
# 我们在 /opt 目录下下载并编译，安装到 /usr/local 后删除源码以减小体积
WORKDIR /opt
RUN git clone https://github.com/Tencent/ncnn.git && \
    cd ncnn && \
    git submodule update --init && \
    mkdir -p build && cd build && \
    # 配置 CMake，关闭 Vulkan 以减少依赖复杂性 (CPU 推理足够)
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DNCNN_VULKAN=OFF \
          -DNCNN_SYSTEM_GLSLANG=OFF \
          -DNCNN_BUILD_TOOLS=ON \
          -DNCNN_BUILD_EXAMPLES=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          .. && \
    make -j$(nproc) && \
    make install && \
    # --- 关键修复：手动处理安装路径偏差 ---
    cp -r install/include/ncnn /usr/local/include/ 2>/dev/null || true && \
    cp -r install/lib/libncnn.a /usr/local/lib/ 2>/dev/null || true && \
    # --- 关键修复：手动编译 onnx2ncnn 工具 ---
    cd ../tools/onnx && \
    protoc --cpp_out=. onnx.proto && \
    g++ -o onnx2ncnn onnx2ncnn.cpp onnx.pb.cc -std=c++11 -O3 \
        -I . -I ../../build/install/include -I ../../build/install/include/ncnn \
        -L ../../build/install/lib -lncnn -lprotobuf -pthread -fopenmp && \
    cp onnx2ncnn /usr/local/bin/ && \
    # 清理源码目录，保持镜像整洁
    cd /opt && rm -rf ncnn

# 4. 安装 Python 依赖 (MuseTalk 导出需要)
# 安装 CPU 版 PyTorch 和其他库
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install diffusers transformers accelerate opencv-python onnx onnxsim huggingface_hub omegaconf soundfile einops

# 5. 配置 SSH 服务和语言环境
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# 6. 设置工作目录 (这里将会挂载你的代码)
WORKDIR /workspace

# 7. 暴露端口
EXPOSE 22

# 8. 启动命令
CMD ["/usr/sbin/sshd", "-D"]