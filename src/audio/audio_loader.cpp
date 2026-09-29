#include <iostream>

#include "audio/audio_loader.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

namespace DigitalHuman {
namespace Audio {

struct AudioLoader::Impl {
    int target_rate; // 目标采样率，如 16000

    explicit Impl(int rate) : target_rate(rate) {}

    bool process(const std::string& filename, std::vector<int16_t>& out_pcm) {
        AVFormatContext* format_ctx = nullptr;
        
        // 1. 打开多媒体文件：探测协议、读取文件头
        if (avformat_open_input(&format_ctx, filename.c_str(), nullptr, nullptr) != 0) {
            return false;
        }

        // 2. 检索流信息：读取部分帧以获取准确的编解码参数（如采样率、时长等）
        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
            avformat_close_input(&format_ctx);
            return false;
        }

        // 3. 筛选音频流：在容器中自动查找最佳的音频轨道索引
        int stream_idx = av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (stream_idx < 0) {
            avformat_close_input(&format_ctx);
            return false;
        }

        // 4. 初始化解码器上下文
        AVCodecParameters* codecpar = format_ctx->streams[stream_idx]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            avformat_close_input(&format_ctx);
            return false;
        }

        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(codec_ctx, codecpar); // 将流参数复制到解码上下文

        // 5. 开启解码器：准备开始处理压缩数据
        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&format_ctx);
            return false;
        }

        // --- 6. 重采样配置 (SwrContext) ---
        // 获取源音频的声道布局
        int64_t in_ch_layout = codec_ctx->channel_layout;
        if (in_ch_layout == 0) {
            in_ch_layout = av_get_default_channel_layout(codec_ctx->channels);
        }

        // 设定目标输出参数：单声道、S16格式（2字节整型）、目标采样率
        int64_t out_ch_layout = AV_CH_LAYOUT_MONO;
        SwrContext* swr_ctx = swr_alloc_set_opts(
            nullptr,
            out_ch_layout, AV_SAMPLE_FMT_S16, target_rate,   // 输出参数
            in_ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate, // 输入参数
            0, nullptr
        );

        if (!swr_ctx || swr_init(swr_ctx) < 0) {
            if (swr_ctx) swr_free(&swr_ctx);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&format_ctx);
            return false;
        }
        // ------------------------------------------

        out_pcm.clear();
        AVPacket* packet = av_packet_alloc(); // 存放压缩数据包
        AVFrame* frame = av_frame_alloc();   // 存放解码后的原始帧

        // 7. 主解码循环：逐包读取 -> 发送解码 -> 接收原始数据
        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == stream_idx) {
                // 向解码器送入压缩包
                if (avcodec_send_packet(codec_ctx, packet) == 0) {
                    // 从解码器提取解码后的原始采样点 (可能有多个Frame)
                    while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                        // A. 计算重采样后的输出样本数（含重采样器内部延迟的样本）
                        int out_samples = av_rescale_rnd(
                            swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
                            target_rate, codec_ctx->sample_rate, AV_ROUND_UP);

                        // B. 准备转换缓冲区
                        std::vector<int16_t> temp_buf(out_samples);
                        uint8_t* out_data_ptr = reinterpret_cast<uint8_t*>(temp_buf.data());
                        
                        // C. 执行重采样转换
                        int converted = swr_convert(swr_ctx, &out_data_ptr, out_samples,
                                                   (const uint8_t**)frame->data, frame->nb_samples);
                        
                        // D. 将转换后的标准化 PCM 数据存入结果集
                        if (converted > 0) {
                            out_pcm.insert(out_pcm.end(), temp_buf.begin(), temp_buf.begin() + converted);
                        }
                    }
                }
            }
            av_packet_unref(packet); // 释放 packet 引用，避免内存泄漏
        }

        // 8. 刷新重采样缓冲区 (Flush)：处理 SwrContext 内部残留的最后几毫秒音频
        int delayed_samples = 0;
        do {
            std::vector<int16_t> flush_buf(1024);
            uint8_t* flush_ptr = reinterpret_cast<uint8_t*>(flush_buf.data());
            // 传入 nullptr 代表尝试冲刷剩余数据
            delayed_samples = swr_convert(swr_ctx, &flush_ptr, 1024, nullptr, 0);
            if (delayed_samples > 0) {
                out_pcm.insert(out_pcm.end(), flush_buf.begin(), flush_buf.begin() + delayed_samples);
            }
        } while (delayed_samples > 0);

        // 9. 资源销毁：遵循“先申请后释放”原则，防止内存泄漏和句柄占用
        av_frame_free(&frame);
        av_packet_free(&packet);
        swr_free(&swr_ctx);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);

        return true;
    }
};

AudioLoader::AudioLoader(int target_sample_rate) 
    : pimpl(std::make_unique<Impl>(target_sample_rate)) {}

AudioLoader::~AudioLoader() = default;

bool AudioLoader::load(const std::string& filename, std::vector<int16_t>& out_pcm) {
    return pimpl->process(filename, out_pcm);
}

int AudioLoader::getTargetSampleRate() const {
    return pimpl->target_rate;
}

} // namespace Audio
} // namespace DigitalHuman