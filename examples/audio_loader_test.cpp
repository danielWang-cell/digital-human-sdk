#include <iostream>
#include <iomanip>

#include "audio/audio_loader.h"

using namespace DigitalHuman::Audio;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./audio_loader_test <audio_path>" << std::endl;
        return -1;
    }

    const std::string filePath = argv[1];
    AudioLoader loader(16000); // 指定数字人常用的 16kHz
    std::vector<int16_t> pcmData;

    std::cout << "[Test] Loading: " << filePath << std::endl;

    if (loader.load(filePath, pcmData)) {
        double duration = static_cast<double>(pcmData.size()) / loader.getTargetSampleRate();
        
        std::cout << ">> Success!" << std::endl;
        std::cout << ">> Sample Count: " << pcmData.size() << std::endl;
        std::cout << ">> Duration:     " << std::fixed << std::setprecision(2) << duration << "s" << std::endl;
        std::cout << ">> Target Rate:  " << loader.getTargetSampleRate() << "Hz" << std::endl;
        
        if (!pcmData.empty()) {
            std::cout << ">> First Sample: " << pcmData[0] << std::endl;
        }
    } else {
        std::cerr << ">> Error: Failed to load or decode audio." << std::endl;
        return -1;
    }

    return 0;
}