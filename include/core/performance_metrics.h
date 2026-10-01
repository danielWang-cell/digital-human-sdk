#pragma once

#include <chrono>
#include <cstdint>
#include <atomic>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <mutex>
#include <sstream>
#include <iomanip>

namespace DigitalHuman::Core {

struct StageMetric {
    std::uint64_t samples = 0;
    double total_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;

    double averageMs() const { return samples ? total_ms / samples : 0.0; }
};

class PerformanceMetrics {
public:
    explicit PerformanceMetrics(bool enabled = true) : enabled_(enabled) {}

    class Scope {
    public:
        Scope(PerformanceMetrics& owner, std::string stage)
            : owner_(owner), stage_(std::move(stage)), start_(Clock::now()),
              enabled_(owner.enabled()) {}
        ~Scope() { if (enabled_) owner_.add(stage_, elapsedMs()); }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    private:
        using Clock = std::chrono::steady_clock;
        PerformanceMetrics& owner_;
        std::string stage_;
        Clock::time_point start_;
        bool enabled_;
        double elapsedMs() const {
            return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
        }
    };

    void add(const std::string& stage, double elapsed_ms) {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto& metric = stages_[stage];
        ++metric.samples;
        metric.total_ms += elapsed_ms;
        if (metric.samples == 1) metric.min_ms = elapsed_ms;
        else metric.min_ms = std::min(metric.min_ms, elapsed_ms);
        metric.max_ms = std::max(metric.max_ms, elapsed_ms);
    }

    void setEnabled(bool enabled) { enabled_.store(enabled, std::memory_order_release); }
    bool enabled() const { return enabled_.load(std::memory_order_acquire); }

    const std::unordered_map<std::string, StageMetric>& stages() const { return stages_; }
    std::unordered_map<std::string, StageMetric> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stages_;
    }

    std::string toJson() const {
        const auto copy = snapshot();
        std::ostringstream out;
        out << "{";
        bool first = true;
        out << std::fixed << std::setprecision(3);
        for (const auto& entry : copy) {
            if (!first) out << ",";
            first = false;
            const auto& metric = entry.second;
            out << "\"" << entry.first << "\":{";
            out << "\"samples\":" << metric.samples
                << ",\"avg_ms\":" << metric.averageMs()
                << ",\"min_ms\":" << metric.min_ms
                << ",\"max_ms\":" << metric.max_ms << "}";
        }
        out << "}";
        return out.str();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stages_.clear();
    }

private:
    std::atomic<bool> enabled_{true};
    mutable std::mutex mutex_;
    std::unordered_map<std::string, StageMetric> stages_;
};

} // namespace DigitalHuman::Core
