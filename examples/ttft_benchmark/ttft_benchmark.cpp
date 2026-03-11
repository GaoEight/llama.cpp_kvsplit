#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

constexpr int CHUNK_SIZE = 128;
// 计时器工具（微秒级）
struct Timer {
    int64_t start;
    int64_t& accumulator;
    
    Timer(int64_t& acc) : accumulator(acc) {
        start = ggml_time_us();
    }
    
    ~Timer() {
        accumulator = ggml_time_us() - start;
    }
};

// 正常 Prefill：一次性处理所有 tokens
int64_t benchmark_normal_prefill(
    llama_context* ctx,
    std::vector<llama_token>& tokens
) {
    int64_t elapsed_us = 0;
    
    // 创建 batch：包含所有 tokens
    llama_batch batch = llama_batch_get_one(
        tokens.data(),      // token 数组
        tokens.size()       // token 数量
    );
    
    {
        Timer timer(elapsed_us);
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "llama_decode failed: %d\n", ret);
            return -1;
        }
        // CPU 后端：decode 返回时计算已完成
    }
    
    return elapsed_us;
}

// 分块 Prefill：将 tokens 分成 chunks 处理
int64_t benchmark_chunked_prefill(
    llama_context* ctx,
    std::vector<llama_token>& tokens,
    int chunk_size = 128
) {
    int64_t elapsed_us = 0;
    int n_past = 0;  // 记录已处理的 token 数
    
    {
        Timer timer(elapsed_us);
        
        // 分块处理
        for (size_t i = 0; i < tokens.size(); i += chunk_size) {
            size_t n_eval = std::min((size_t)chunk_size, tokens.size() - i);
            
            // 创建当前 chunk 的 batch
            llama_batch batch = llama_batch_get_one(
                tokens.data() + i,  // 当前 chunk 的起始位置
                n_eval               // 当前 chunk 的大小
            );
            
            // 执行 decode
            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                fprintf(stderr, "llama_decode failed at chunk %zu: %d\n", 
                        i / chunk_size, ret);
                return -1;
            }
            
            n_past += n_eval;
        }
    }
    
    return elapsed_us;
}

// 清空 KV cache（测试之间需要重置）
void reset_kv_cache(llama_context* ctx) {
    llama_memory_clear(llama_get_memory(ctx), true);
}

static void print_usage(int argc, char ** argv) {
    printf("\nexample usage:\n");
    printf("    %s -m model.gguf [-c n_ctx] [--chunk-size size] [--prompt text]\n", argv[0]);
    printf("\noptions:\n");
    printf("    -m, --model PATH      Path to model file (required)\n");
    printf("    -c, --ctx N           Context size (default: 4096)\n");
    printf("    --chunk-size N        Chunk size for chunked prefill (default: 256)\n");
    printf("    --prompt TEXT         Prompt text (default: long test prompt)\n");
    printf("\n");
}

int main(int argc, char** argv) {
    // 参数默认值
    std::string model_path;
    std::string prompt;
    int n_ctx = 4096;
    int chunk_size = 256;
    
    // 生成一个长测试 prompt（如果不提供）
    std::string default_prompt = 
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "in contrast to the natural intelligence displayed by humans and animals. "
        "Leading AI textbooks define the field as the study of intelligent agents: "
        "any device that perceives its environment and takes actions that maximize "
        "its chance of successfully achieving its goals. Colloquially, the term "
        "artificial intelligence is often used to describe machines that mimic "
        "cognitive functions that humans associate with the human mind, such as "
        "learning and problem solving. As machines become increasingly capable, "
        "tasks considered to require intelligence are often removed from the "
        "definition of AI, a phenomenon known as the AI effect. A quip in Tesler's "
        "Theorem says AI is whatever hasn't been done yet. For instance, optical "
        "character recognition is frequently excluded from things considered to be "
        "AI, having become a routine technology. Modern machine capabilities "
        "generally classified as AI include successfully understanding human speech, "
        "competing at the highest level in strategic game systems, autonomously "
        "operating cars, intelligent routing in content delivery networks, and "
        "military simulations. Artificial intelligence was founded as an academic "
        "discipline in 1956, and in the years since has experienced several waves "
        "of optimism, followed by disappointment and the loss of funding known as "
        "AI winter, followed by new approaches, success and renewed funding. For "
        "most of its history, AI research has been divided into subfields that "
        "often fail to communicate with each other. These sub-fields are based on "
        "technical considerations, such as particular goals, the use of particular "
        "tools, or deep philosophical differences. Subfields have also been based "
        "on social factors particular institutions or the work of particular "
        "researchers. The traditional problems or goals of AI research include "
        "reasoning, knowledge representation, planning, learning, natural language "
        "processing, perception and the ability to move and manipulate objects. "
        "General intelligence is among the field's long-term goals. Approaches "
        "include statistical methods, computational intelligence, and traditional "
        "symbolic AI. Many tools are used in AI, including versions of search and "
        "mathematical optimization, artificial neural networks, and methods based "
        "on statistics, probability and economics. The AI field draws upon computer "
        "science, information engineering, mathematics, psychology, linguistics, "
        "philosophy, and many other fields. The field was founded on the assumption "
        "that human intelligence can be so precisely described that a machine can "
        "be made to simulate it. This raises philosophical arguments about the "
        "nature of the mind and the ethics of creating artificial beings endowed "
        "with human-like intelligence. These issues have been explored by myth, "
        "fiction and philosophy since antiquity. Some people also consider AI to "
        "be a danger to humanity if it progresses unabated. Others believe that AI, "
        "unlike previous technological revolutions, will create a risk of mass "
        "unemployment. In the twenty-first century, AI techniques have experienced "
        "a resurgence following concurrent advances in computer power, large "
        "amounts of data, and theoretical understanding; and AI techniques have "
        "become an essential part of the technology industry, helping to solve "
        "many challenging problems in computer science, software engineering and "
        "operations research.";
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else if (arg == "-c" || arg == "--ctx") {
            if (i + 1 < argc) {
                n_ctx = std::stoi(argv[++i]);
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else if (arg == "--chunk-size") {
            if (i + 1 < argc) {
                chunk_size = std::stoi(argv[++i]);
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else if (arg == "--prompt") {
            if (i + 1 < argc) {
                prompt = argv[++i];
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return 0;
        } else {
            // 未知参数
            printf("Unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            return 1;
        }
    }
    
    // 检查必要参数
    if (model_path.empty()) {
        printf("Error: Model path is required\n");
        print_usage(argc, argv);
        return 1;
    }
    
    // 使用默认 prompt
    if (prompt.empty()) {
        prompt = default_prompt;
        printf("Using default test prompt (length: %zu chars)\n", prompt.size());
    }
    
    printf("\n");
    printf("===============================================\n");
    printf("TTFT Benchmark: Normal vs Chunked Prefill\n");
    printf("===============================================\n");
    printf("Model:         %s\n", model_path.c_str());
    printf("Context size:  %d\n", n_ctx);
    printf("Chunk size:    %d\n", chunk_size);
    printf("\n");
    
    // 1. 初始化后端（自动加载所有可用后端）
    printf("Loading backends...\n");
    ggml_backend_load_all();
    
    // 2. 加载模型
    printf("Loading model...\n");
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // 强制使用 CPU
    
    llama_model* model = llama_model_load_from_file(
        model_path.c_str(), 
        model_params
    );
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_path.c_str());
        return 1;
    }
    printf("Model loaded successfully\n");
    
    // 3. 创建 context
    printf("Creating context...\n");
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;     // 允许处理大量 tokens
    ctx_params.n_ubatch = 256;      // 微 batch 大小
    ctx_params.no_perf = false;     // 启用性能统计
    
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    printf("Context created successfully\n\n");
    
    // 4. Tokenize
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const int n_prompt =  
        -llama_tokenize(vocab, prompt.c_str(), 
        prompt.size(), NULL, 0, 
        true, true);

    
    std::vector<llama_token> tokens(n_prompt);
    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        prompt.size(),
        tokens.data(),
        tokens.size(),
        true,   // add_special
        true    // parse_special
    );
    
    if (n_tokens < 0) {
        // 需要更大的缓冲区
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt.size(),
            tokens.data(),
            tokens.size(),
            true,
            true
        );
    }
    tokens.resize(n_tokens);
    
    printf("Tokenization complete:\n");
    printf("  Input characters: %zu\n", prompt.size());
    printf("  Tokens:           %d\n", n_tokens);
    printf("  Tokens/chars:     %.2f\n", (float)n_tokens / prompt.size());
    printf("\n");
    
    if (n_tokens == 0) {
        fprintf(stderr, "Error: No tokens generated from prompt\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    // 5. 测试正常 Prefill
    printf("===============================================\n");
    printf("Test 1: Normal Prefill (single batch)\n");
    printf("===============================================\n");
    printf("Processing %d tokens in one batch...\n", n_tokens);
    
    int64_t time_normal = benchmark_normal_prefill(ctx, tokens);
    if (time_normal < 0) {
        fprintf(stderr, "Error: Normal prefill failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    printf("  TTFT: %.3f ms (%.3f ms/token)\n", 
           time_normal / 1000.0, 
           time_normal / 1000.0 / n_tokens);
    printf("\n");
    
    // 6. 重置 KV cache
    printf("Resetting KV cache...\n");
    reset_kv_cache(ctx);
    printf("KV cache cleared\n\n");
    
    // 7. 测试分块 Prefill
    int num_chunks = (n_tokens + chunk_size - 1) / chunk_size;
    
    printf("===============================================\n");
    printf("Test 2: Chunked Prefill (%d chunks)\n", num_chunks);
    printf("===============================================\n");
    printf("Processing %d tokens in chunks of %d...\n", n_tokens, chunk_size);
    
    int64_t time_chunked = benchmark_chunked_prefill(ctx, tokens, chunk_size);
    if (time_chunked < 0) {
        fprintf(stderr, "Error: Chunked prefill failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    printf("  TTFT: %.3f ms (%.3f ms/token)\n", 
           time_chunked / 1000.0, 
           time_chunked / 1000.0 / n_tokens);
    printf("  Chunks: %d\n", num_chunks);
    printf("\n");
    
    // 8. 结果对比
    double diff_ms = (time_chunked - time_normal) / 1000.0;
    double diff_percent = (time_chunked - time_normal) * 100.0 / time_normal;
    
    printf("===============================================\n");
    printf("Results Comparison\n");
    printf("===============================================\n");
    printf("Normal Prefill:   %10.3f ms  (%8.3f ms/token)\n", 
           time_normal / 1000.0, 
           time_normal / 1000.0 / n_tokens);
    printf("Chunked Prefill:  %10.3f ms  (%8.3f ms/token)\n", 
           time_chunked / 1000.0, 
           time_chunked / 1000.0 / n_tokens);
    printf("\n");
    printf("Difference:       %10.3f ms  (%+.2f%%)\n", 
           diff_ms, 
           diff_percent);
    printf("\n");
    
    // 结论
    printf("Conclusion:\n");
    if (std::abs(diff_percent) < 1.0) {
        printf("  The TTFT difference is negligible (< 1%%).\n");
        printf("  Chunked prefill has minimal overhead on CPU backend.\n");
    } else if (diff_percent > 0) {
        printf("  Chunked prefill is slower by %.2f%%.\n", diff_percent);
        printf("  This may be due to function call overhead or cache effects.\n");
    } else {
        printf("  Chunked prefill is faster by %.2f%%.\n", -diff_percent);
        printf("  This could be due to better cache utilization.\n");
    }
    printf("\n");
    
    // 9. 清理
    llama_free(ctx);
    llama_model_free(model);
    
    printf("Benchmark completed.\n");
    
    return 0;
}
