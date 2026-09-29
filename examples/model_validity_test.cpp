#include <ncnn/net.h>
#include <ncnn/mat.h>
#include <ncnn/layer.h>
#include <ncnn/blob.h> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

// Helper: Find blob index using public API
int get_blob_index(const ncnn::Net& net, const std::string& name) {
    const std::vector<ncnn::Blob>& blobs = net.blobs();
    for(size_t i=0; i<blobs.size(); i++) {
        if (blobs[i].name == name) return (int)i;
    }
    return -1;
}

// Helper: Random float
void fill_random_float(ncnn::Mat& m, float mean = 0.0f, float stddev = 1.0f) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(mean, stddev);
    float* ptr = (float*)m.data;
    int size = m.w * m.h * m.c;
    for (int i = 0; i < size; i++) ptr[i] = dist(rng);
}

struct NodeInfo {
    std::string type;
    std::string name; // layer name
    std::vector<std::string> bottoms;
    std::vector<std::string> tops;
};

struct InputTrace {
    std::string input_name;
    std::string role_guess;
    std::string target_layer_name;
    std::string target_layer_type;
    std::string target_blob_name; 
    std::string trace_log; // Why it stopped
};

// 1. Deep Structure Parser: Trace through Split/Reshape to find computation
std::vector<InputTrace> trace_inputs(const std::string& param_path) {
    std::ifstream fin(param_path);
    std::vector<InputTrace> results;
    if (!fin.is_open()) return results;

    // Build Graph
    std::map<std::string, NodeInfo> layer_map; 
    std::map<std::string, std::vector<NodeInfo*>> consumers; 
    std::vector<NodeInfo> all_layers; 
    std::vector<std::string> input_blobs;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.length() < 5) continue;
        std::stringstream ss(line);
        std::string type, name;
        ss >> type >> name;

        if (type == "Input") {
            int bottom_c, top_c; ss >> bottom_c >> top_c;
            std::string blob_name; ss >> blob_name;
            input_blobs.push_back(blob_name);
        } else if (!isdigit(type[0])) {
            NodeInfo node;
            node.type = type;
            node.name = name;
            int bottom_c, top_c; ss >> bottom_c >> top_c;
            for(int i=0; i<bottom_c; i++) {
                std::string b; ss >> b; node.bottoms.push_back(b);
            }
            for(int i=0; i<top_c; i++) {
                std::string t; ss >> t; node.tops.push_back(t);
            }
            all_layers.push_back(node);
        }
    }
    
    // Index consumers
    for(size_t i=0; i<all_layers.size(); i++) {
        for(const auto& b : all_layers[i].bottoms) {
             consumers[b].push_back(&all_layers[i]);
        }
    }

    // BFS Trace Function
    auto find_compute_layer = [&](std::string start_blob, std::string& log) -> NodeInfo* {
        std::vector<std::string> q;
        q.push_back(start_blob);
        std::set<std::string> visited;
        visited.insert(start_blob);

        int steps = 0;
        int max_steps = 50; 

        while(!q.empty() && steps < max_steps) { 
            std::string curr_blob = q.front();
            q.erase(q.begin());

            if (consumers.find(curr_blob) != consumers.end()) {
                for (NodeInfo* layer : consumers[curr_blob]) {
                    std::string t = layer->type;
                    
                    // 1. Computation Layers (Stop here)
                    if (t == "Convolution" || t == "ConvolutionDepthWise" || 
                        t == "InnerProduct" || t == "MatMul" || t == "Gemm" || 
                        t == "Embed" || t == "GroupNorm" || t == "LayerNorm" ||
                        t == "MultiHeadAttention") {
                        return layer;
                    }
                    
                    // 2. Passthrough Layers (Continue tracing)
                    // Added Tile, Broadcast, Expand, Reorg, ExpandDims, Unsqueeze
                    // [Fix] Added "Noop" to fix Timestep tracing
                    if (t == "Split" || t == "Reshape" || t == "Permute" || 
                        t == "Flatten" || t == "Slice" || t == "Tile" || 
                        t == "Expand" || t == "BroadCast" || t == "Reorg" ||
                        t == "Padding" || t == "Crop" || t == "Concat" ||
                        t == "Dropout" || t == "Cast" || t == "ExpandDims" || 
                        t == "Unsqueeze" || t == "Squeeze" || t == "Noop") {
                        
                        for(const auto& next_blob : layer->tops) {
                            if(visited.find(next_blob) == visited.end()) {
                                visited.insert(next_blob);
                                q.push_back(next_blob);
                            }
                        }
                    } else {
                        // Unknown layer type encountered
                        log += "Stopped at unknown/unhandled layer type: " + t + " (" + layer->name + "); ";
                    }
                }
            } else {
                log += "Blob " + curr_blob + " has no consumers; ";
            }
            steps++;
        }
        if (steps >= max_steps) log += "Max BFS depth reached; ";
        return nullptr;
    };

    for (const auto& in_name : input_blobs) {
        InputTrace trace;
        trace.input_name = in_name;
        
        NodeInfo* target = find_compute_layer(in_name, trace.trace_log);
        
        if (target) {
            trace.target_layer_name = target->name;
            trace.target_layer_type = target->type;
            if(!target->tops.empty()) trace.target_blob_name = target->tops[0];
            
            // Guess Role based on deep target
            if (target->type == "Convolution") trace.role_guess = "Latent";
            else if (target->type == "InnerProduct" || target->type == "MatMul" || 
                     target->type == "MultiHeadAttention") trace.role_guess = "Audio";
            else if (target->type == "Embed") trace.role_guess = "Timestep";
            else trace.role_guess = "Unknown";
        } else {
            trace.target_layer_type = "None";
            trace.target_blob_name = "None";
            trace.role_guess = "Unknown";
        }
        results.push_back(trace);
    }
    return results;
}

// 2. Surgical Probe: Run inference
int probe_layer_run(const std::string& param_path, 
                    const std::string& bin_path,
                    const std::string& input_name,
                    const std::string& target_output_name,
                    int w, int h, int c)
{
    ncnn::Net net;
    net.opt.use_local_pool_allocator = false; 
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = false;

    if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0) return -9;

    int in_idx = get_blob_index(net, input_name);
    int out_idx = get_blob_index(net, target_output_name);
    
    if (in_idx == -1 || out_idx == -1) return -8;

    ncnn::Extractor ex = net.create_extractor();
    
    ncnn::Mat in_mat(w, h, c);
    fill_random_float(in_mat);
    
    ex.input(in_idx, in_mat);
    
    ncnn::Mat out_mat;
    int ret = ex.extract(out_idx, out_mat);
    
    if (ret != 0) return ret; 
    if (out_mat.empty()) return -2;

    double sum = 0;
    const float* ptr = (const float*)out_mat.data;
    int total = out_mat.w * out_mat.h * out_mat.c;
    for(int i=0; i<total; i++) sum += std::abs(ptr[i]);
    
    std::cout << "      [Res] Shape:" << out_mat.w << "x" << out_mat.h << "x" << out_mat.c 
              << " Mean:" << sum/total << " ";

    if (sum < 1e-6) return 0;
    return 1;
}

// 3. Combined Inference (The Final Test)
int run_combined_test(const std::string& param_path, 
                      const std::string& bin_path,
                      const std::string& lat_name, int lat_c,
                      const std::string& aud_name, int aud_w, int aud_h,
                      const std::string& time_name,
                      const std::string& out_name)
{
    ncnn::Net net;
    net.opt.use_local_pool_allocator = false; 
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = false;
    net.opt.use_fp16_arithmetic = false;

    if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0) return -9;

    ncnn::Extractor ex = net.create_extractor();

    // 1. Latent
    ncnn::Mat m_lat(32, 32, lat_c); fill_random_float(m_lat);
    ex.input(lat_name.c_str(), m_lat);

    // 2. Audio
    ncnn::Mat m_aud(aud_w, aud_h, 1); fill_random_float(m_aud);
    ex.input(aud_name.c_str(), m_aud);

    // 3. Timestep (Scalar)
    ncnn::Mat m_time(1); m_time.fill(0.f);
    ex.input(time_name.c_str(), m_time);

    // 4. Extract
    ncnn::Mat out;
    int ret = ex.extract(out_name.c_str(), out);
    if (ret != 0) return ret;
    
    double sum = 0;
    int total = out.w * out.h * out.c;
    const float* p = (const float*)out.data;
    for(int i=0; i<total; i++) sum += std::abs(p[i]);

    if (sum < 1e-6) return 0; // Zero
    return 1; // Success
}

// Helper: Find output blob index using public API
int find_output_index_from_net(const ncnn::Net& net, std::string& out_name_str) {
    const std::vector<ncnn::Layer*>& layers = net.layers();
    const std::vector<ncnn::Blob>& blobs = net.blobs();
    if (layers.empty()) return -1;
    ncnn::Layer* last_layer = layers.back();
    if (last_layer->tops.size() > 0) {
        int blob_index = last_layer->tops[0];
        if (blob_index >= 0 && (size_t)blob_index < blobs.size()) {
            out_name_str = blobs[blob_index].name;
            return blob_index;
        }
    }
    return -1;
}

int main(int argc, char** argv) {
    std::string model_dir = "/workspace/models";
    if (argc >= 2) model_dir = argv[1];
    std::string param_path = model_dir + "/musetalk_unet.ncnn.param";
    std::string bin_path = model_dir + "/musetalk_unet.ncnn.bin";

    std::cout << "=== MuseTalk Deep Trace Debugger v2.2 (Fixes Noop & Audio) ===" << std::endl;

    // 1. Trace Structure
    auto traces = trace_inputs(param_path);
    if (traces.empty()) { std::cerr << "Param parsing failed.\n"; return -1; }

    std::string lat_name, aud_name, time_name;
    
    std::cout << "\n[Deep Trace Map]" << std::endl;
    for (const auto& t : traces) {
        if (t.target_blob_name != "None") {
            std::cout << "Input: " << t.input_name << " (" << t.role_guess << ") >>> " 
                      << t.target_layer_type << " (" << t.target_layer_name << ") -> Blob [" << t.target_blob_name << "]" << std::endl;
        } else {
            std::cout << "Input: " << t.input_name << " (Trace Failed) >>> Log: " << t.trace_log << std::endl;
        }
        
        if (t.role_guess == "Latent") lat_name = t.input_name;
        if (t.role_guess == "Audio") aud_name = t.input_name;
        if (t.role_guess == "Timestep") time_name = t.input_name;
    }

    // Fallback if role guessing failed
    if (lat_name.empty()) lat_name = "in0";
    if (aud_name.empty()) aud_name = "in2";
    if (time_name.empty()) time_name = "in1";

    // -------------------------------------------------------------------------
    // TEST 1: Timestep Probe (Now fixed with Noop)
    // -------------------------------------------------------------------------
    std::cout << "\n=== 1. Probing Timestep (Scalar) ===" << std::endl;
    for (const auto& t : traces) {
        if (t.role_guess == "Timestep" && t.target_blob_name != "None") {
             std::cout << "  Attempt scalar: ";
             int r = probe_layer_run(param_path, bin_path, t.input_name, t.target_blob_name, 1, 1, 1);
             std::cout << (r==1 ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << std::endl;
        }
    }

    // -------------------------------------------------------------------------
    // TEST 2: Combined Inference (The Solution)
    // -------------------------------------------------------------------------
    std::cout << "\n=== 2. Combined Inference (Full Net) ===" << std::endl;
    std::cout << "Using fixed roles: Latent=" << lat_name << " (8ch), Timestep=" << time_name << ", Audio=" << aud_name << std::endl;
    
    // Get output name
    ncnn::Net tmp_net; tmp_net.load_param(param_path.c_str());
    std::string out_name; find_output_index_from_net(tmp_net, out_name);
    
    struct AudCfg { int w, h; };
    AudCfg cfgs[] = {{384, 50}, {50, 384}, {768, 50}, {50, 768}, {512, 50}, {50, 512}};

    for (const auto& c : cfgs) {
        std::cout << "  Testing Audio " << c.w << "x" << c.h << "... ";
        int r = run_combined_test(param_path, bin_path, lat_name, 8, aud_name, c.w, c.h, time_name, out_name);
        
        if (r == 1) {
            std::cout << "\033[32m[SUCCESS] Non-Zero Output!\033[0m" << std::endl;
            std::cout << "\n🎉 FINAL SOLUTION DETECTED:" << std::endl;
            std::cout << "Latent: 8 Channels" << std::endl;
            std::cout << "Audio:  " << c.w << " x " << c.h << " (ncnn Mat w x h)" << std::endl;
            return 0;
        } else {
            std::cout << "[FAIL " << r << "]" << std::endl;
        }
    }
    
    std::cout << "\n[FAIL] All combined tests failed." << std::endl;

    return 0;
}