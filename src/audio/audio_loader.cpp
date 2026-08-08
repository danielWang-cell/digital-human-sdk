#include "audio/audio_loader.h"
#include <iostream>
#include <algorithm>

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
    // 目标采样率 
    int target_rate = 16000;

    explicit Impl(int rate) : target_rate(rate) {}

    bool loadInt16(const std::string& filename, std::vector<int16_t>& out_pcm) {
        out_pcm.clear();

        AVFormatContext* format_ctx = nullptr;
        if (avformat_open_input(&format_ctx, filename.c_str(), nullptr, nullptr) != 0) {
            std::cerr << "[AudioLoader] Failed to open input: " << filename << std::endl;
            return false;
        }

        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
            std::cerr << "[AudioLoader] Failed to find stream info." << std::endl;
            avformat_close_input(&format_ctx);
            return false;
        }

        int stream_idx = av_find_best_stream(
            format_ctx,
            AVMEDIA_TYPE_AUDIO,
            -1,
            -1,
            nullptr,
            0
        );

        if (stream_idx < 0) {
            std::cerr << "[AudioLoader] No audio stream found." << std::endl;
            avformat_close_input(&format_ctx);
            return false;
        }
        
        AVCodecParameters* codecpar = format_ctx->streams[stream_idx]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            std::cerr << "[AudioLoader] Decoder not found." << std::endl;
            avformat_close_input(&format_ctx);
            return false;
        }

        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            avformat_close_input(&format_ctx);
            return false;
        }
        
        if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&format_ctx);
            return false;
        }

        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            std::cerr << "[AudioLoader] Failed to open decoder." << std::endl;
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&format_ctx);
            return false;
        }

        int64_t in_ch_layout = codec_ctx->channel_layout;
        if (in_ch_layout == 0) {
            in_ch_layout = av_get_default_channel_layout(codec_ctx->channels);
        }
        
        const int64_t out_ch_layout = AV_CH_LAYOUT_MONO;
        
        SwrContext* swr_ctx = swr_alloc_set_opts(
            nullptr,
            out_ch_layout,
            AV_SAMPLE_FMT_S16,
            target_rate,
            in_ch_layout,
            codec_ctx->sample_fmt,
            codec_ctx->sample_rate,
            0,
            nullptr
        );

        if (!swr_ctx || swr_init(swr_ctx) < 0) {
            std::cerr << "[AudioLoader] Failed to init resampler." << std::endl;
            if (swr_ctx) {
                swr_free(&swr_ctx);
            }
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&format_ctx);
            return false;
        }

        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        if (!packet || !frame) {
            if (frame) {
                av_frame_free(&frame);
            }
            if (packet) {
                av_packet_free(&packet);
            }
            swr_free(&swr_ctx);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&format_ctx);
            return false;
        }

        auto receiveDecodedFrames = [&]() {
            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                int out_samples = av_rescale_rnd(
                    swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
                    target_rate,
                    codec_ctx->sample_rate,
                    AV_ROUND_UP
                );
                std::vector<int16_t> temp(out_samples);
                uint8_t* out_data = reinterpret_cast<uint8_t*>(temp.data());

                int converted = swr_convert(
                    swr_ctx,
                    &out_data,
                    out_samples,
                    const_cast<const uint8_t**>(frame->data),
                    frame->nb_samples
                );

                if (converted > 0) {
                    out_pcm.insert(out_pcm.end(), temp.begin(), temp.begin() + converted);
                }

                av_frame_unref(frame);
            }
        };

        // 主解码循环
        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == stream_idx) {
                if (avcodec_send_packet(codec_ctx, packet) == 0) {
                    receiveDecodedFrames();
                }
            }
            av_packet_unref(packet);
        }

        // Flush 解码器
        avcodec_send_packet(codec_ctx, nullptr);
        receiveDecodedFrames();

        // Flush 重采样器残余数据
        while (true) {
            std::vector<int16_t> temp(1024);
            uint8_t* out_data = reinterpret_cast<uint8_t*>(temp.data());

            int converted = swr_convert(swr_ctx, &out_data, 1024, nullptr, 0);
            if (converted <= 0) {
                break;
            }
            out_pcm.insert(out_pcm.end(), temp.begin(), temp.begin() + converted);
        }

        // 清理 FFmpeg 资源
        av_frame_free(&frame);
        av_packet_free(&packet);
        swr_free(&swr_ctx);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);

        return !out_pcm.empty();
    }

    // int16 线性归一化转 float [-1.0, 1.0]
    static void int16ToFloat(const std::vector<int16_t>& in_pcm, std::vector<float>& out_pcm) {
        out_pcm.resize(in_pcm.size());
        for (size_t i = 0; i < in_pcm.size(); ++i) {
            out_pcm[i] = static_cast<float>(in_pcm[i]) / 32768.0f;
        }
    }
};

// ---------------------------------------------------------------------------
// AudioLoader 外部公有 API 实现
// ---------------------------------------------------------------------------

AudioLoader::AudioLoader(int target_sample_rate)
    : pImpl(std::make_unique<Impl>(target_sample_rate)) {}

AudioLoader::~AudioLoader() = default;

bool AudioLoader::load(const std::string& filename, std::vector<int16_t>& out_pcm) {
    return pImpl->loadInt16(filename, out_pcm);
}

bool AudioLoader::loadFloat(const std::string& filename, std::vector<float>& out_pcm) {
    std::vector<int16_t> pcm16;
    if (!pImpl->loadInt16(filename, pcm16)) {
        return false;
    }
    Impl::int16ToFloat(pcm16, out_pcm);
    return true;
}

int AudioLoader::getTargetSampleRate() const {
    return pImpl->target_rate;
}

} // namespace Audio
} // namespace DigitalHuman