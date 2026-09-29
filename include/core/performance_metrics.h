#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <utility>

namespace DigitalHuman::Core {

struct StageMetric {
    std::uint64_t samples = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;

    double averageMs() const { return samples ? total_ms / samples : 0.0; }
};

class PerformanceMetrics {
public:
    class Scope {
    public:
        Scope(PerformanceMetrics& owner, std::string stage)
            : owner_(owner), stage_(std::move(stage)), start_(Clock::now()) {}
        ~Scope() { owner_.add(stage_, elapsedMs()); }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    private:
        using Clock = std::chrono::steady_clock;
        PerformanceMetrics& owner_;
        std::string stage_;
        Clock::time_point start_;
        double elapsedMs() const {
            return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
        }
    };

    void add(const std::string& stage, double elapsed_ms) {
        auto& metric = stages_[stage];
        ++metric.samples;
        metric.total_ms += elapsed_ms;
        metric.max_ms = std::max(metric.max_ms, elapsed_ms);
    }

    const std::unordered_map<std::string, StageMetric>& stages() const { return stages_; }
    void reset() { stages_.clear(); }

private:
    std::unordered_map<std::string, StageMetric> stages_;
};

} // namespace DigitalHuman::Core
