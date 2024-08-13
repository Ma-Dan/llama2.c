#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/time.h>

#include "base64.h"
#include "tiktoken.h"
#include <onnxruntime_cxx_api.h>

using namespace std;

const int vocab_size = 151936;

float temp = 1, topp = 0.9;
const int topk = 300;

static const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

class QwenTokenizer {
  public:

    QwenTokenizer(const std::string & tiktoken_path);

    auto encode(const std::string &text, int max_length) const -> std::vector<int>;

    auto decode(const std::vector<int> &ids) const -> std::string;

    auto encode_history(const std::vector<std::string> &history, int max_length) const -> std::vector<int>;

    auto build_prompt(const std::vector<std::string> &history) const -> std::string;

    auto is_special_id(int id) const -> bool;

    tiktoken::tiktoken tokenizer;
    int eos_token_id;
    int im_start_id;
    int im_end_id;
};

static std::pair<std::string, int> _parse(const std::string &line) {
  auto pos = line.find(" ");
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid encoder line: " + line);
  }

  auto token = base64::decode({line.data(), pos});
  int rank = 0;
  try {
    rank = std::stoul(line.substr(pos + 1));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid encoder rank: " + line);
  }

  return {std::move(token), rank};
}

QwenTokenizer::QwenTokenizer(const std::string & tiktoken_path) {
  std::ifstream file(tiktoken_path);
  if (!file) {
    throw std::runtime_error("failed to open encoder file: " + tiktoken_path);
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  std::string line;
  while (std::getline(file, line)) {
    auto [token, rank] = _parse(line);

    if (!encoder.emplace(std::move(token), rank).second) {
      throw std::runtime_error("duplicate item: " + line);
    }
  }

  std::vector<std::string> special_tokens_s{"<|endoftext|>", "<|im_start|>", "<|im_end|>"};
  char buffer[14];
  for (size_t i = 0; i < 205; i++) {
    snprintf(buffer, 14, "<|extra_%zu|>", i);
    special_tokens_s.push_back(buffer);
  }
  size_t encoder_size = encoder.size();
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  special_tokens.reserve(special_tokens_s.size());
  for (size_t i = 0; i < special_tokens_s.size(); i++) {
    special_tokens[special_tokens_s[i]] = encoder_size + i;
  }

  tokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
  eos_token_id = 151643;
  im_start_id = 151644;
  im_end_id = 151645;
}

auto QwenTokenizer::encode(const std::string &text, int max_length) const -> std::vector<int> {
  auto ids = tokenizer.encode(text);
  if ((int)ids.size() > max_length) {
    ids.erase(ids.begin(), ids.end() - max_length);
  }
  return ids;
}

auto QwenTokenizer::decode(const std::vector<int> &ids) const -> std::string {
  std::vector<int> normal_ids(ids);
  normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }),
                   normal_ids.end());
  auto text = tokenizer.decode(normal_ids);
  return text;
}

auto QwenTokenizer::is_special_id(int id) const -> bool {
  return id == eos_token_id || id == im_start_id || id == im_end_id;
}

int sample(const std::vector<float>& logits, float temp, float topp, int topk) {
    // return std::max_element(logits.begin(), logits.end()) - logits.begin();

    assert(logits.size() == vocab_size);

    if (fabsf(temp) < 1e-8)
        return std::max_element(logits.begin(), logits.end()) - logits.begin();

    struct timeval tv;
    gettimeofday(&tv, NULL);
    std::mt19937_64 rng(tv.tv_usec/100);  // haha
    std::uniform_real_distribution<float> dist(0, 1);

    std::vector<std::pair<float, int>> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) probs[i] = {logits[i] / temp, i};
    std::sort(probs.begin(), probs.end(),
              std::greater<std::pair<float, int>>());
    while (probs.size() > topk) probs.pop_back();

    // softmax
    auto maximum = probs[0].first;
    std::transform(probs.begin(), probs.end(), probs.begin(),
                   [maximum](auto x) {
                       return std::make_pair(expf(x.first - maximum), x.second);
                   });
    auto sum = std::accumulate(probs.begin(), probs.end(), 0.0f,
                               [](auto x, auto y) { return x + y.first; });
    std::transform(probs.begin(), probs.end(), probs.begin(), [sum](auto x) {
        return std::make_pair(x.first / sum, x.second);
    });

    sum = 0;
    int last = 0;
    for (int i = 0; i < (int)probs.size(); i++) {
        sum += probs[i].first;
        last = i;
        if (sum > topp) break;
    }

    float r = dist(rng) * sum;
    sum = 0;
    for (int i = 0; i <= last; i++) {
        sum += probs[i].first;
        if (sum > r) return probs[i].second;
    }
    return probs[last].second;
}

int build_transformer(const char* checkpoint, void** p);
void run_transformer(int token, float* embedding, int pos, float* logits, void* p);
void free_transformer(void* p);


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL PROMPT TEMPERATURE"
                  << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    std::string tokenizer_path = "qwen.tiktoken";
    temp = std::atof(argv[3]);

    // 加载tokenizer
    auto tokenizer = std::make_unique<QwenTokenizer>(tokenizer_path);

    // 加载模型
    void* transformer;
    int seq_len = build_transformer(model_path.c_str(), &transformer);

    // 使用onnx运行encoder和projector
    Ort::SessionOptions session_option;
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "qwen_asr" };
    Ort::Session session_encoder{ nullptr };
    Ort::Session session_projector{ nullptr };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Onnx whisper encoder
    vector<const char*> encoder_input_names{ "mel" };
    vector<const char*> encoder_output_names{ "audio_features" };

    std::array<float, 1 * 80 * 3000> encoder_input_;
    std::array<float, 1 * 1500 * 384> encoder_output_;
    std::array<int64_t, 3> encoder_input_shape_{ 1, 80, 3000 };
    std::array<int64_t, 3> encoder_output_shape_{ 1, 1500, 384 };

    Ort::Value encoder_input_tensor_{ nullptr };
    Ort::Value encoder_output_tensor_{ nullptr };

    // Onnx projector
    vector<const char*> projector_input_names{ "input" };
    vector<const char*> projector_output_names{ "output" };

    std::array<float, 1 * 300 * 896> projector_output_;
    std::array<int64_t, 3> projector_input_shape_{ 1, 1500, 384 };
    std::array<int64_t, 3> projector_output_shape_{ 1, 300, 896 };

    Ort::Value projector_input_tensor_{ nullptr };
    Ort::Value projector_output_tensor_{ nullptr };

    // 加载mel
    FILE *finput = fopen("mel_float.bin", "rb");
    fread(encoder_input_.data(), sizeof(float), 80 * 3000, finput);
    fclose(finput);

    // 计算encoder
    encoder_input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, encoder_input_.data(), encoder_input_.size(), encoder_input_shape_.data(), encoder_input_shape_.size());
    encoder_output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, encoder_output_.data(), encoder_output_.size(), encoder_output_shape_.data(), encoder_output_shape_.size());

    session_encoder = Ort::Session(env, "encoder.onnx", session_option);
    session_encoder.Run(Ort::RunOptions{ nullptr }, encoder_input_names.data(), &encoder_input_tensor_, 1, encoder_output_names.data(), &encoder_output_tensor_, 1);

    // 计算projector
    projector_input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, encoder_output_.data(), encoder_output_.size(), projector_input_shape_.data(), projector_input_shape_.size());
    projector_output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, projector_output_.data(), projector_output_.size(), projector_output_shape_.data(), projector_output_shape_.size());

    session_projector = Ort::Session(env, "projector.onnx", session_option);
    session_projector.Run(Ort::RunOptions{ nullptr }, projector_input_names.data(), &projector_input_tensor_, 1, projector_output_names.data(), &projector_output_tensor_, 1);

    // speech + prompt embedding
    vector<int> tokens(seq_len, 0);
    int num_prompt_tokens = 324;

    string transcribe_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nTrascribe the speech<|im_end|>\n<|im_start|>assistant\n";

    // tokenize prompt
    auto tokens_prompt = tokenizer->encode(transcribe_prompt, 32);
    for(int i=0; i<tokens_prompt.size(); i++) {
      tokens[300+i] = tokens_prompt[i];
    }

    int pos = 0;
    vector<float> logits(vocab_size, 0);

    struct timeval tvs, tve;
    gettimeofday(&tvs, NULL);

    for (; pos < 512; pos++) {
      if(pos < 300) {
        run_transformer(-1, projector_output_.data(), pos, logits.data(), transformer);
      } else {
        run_transformer(tokens[pos], NULL, pos, logits.data(), transformer);
      }

      if(pos >= num_prompt_tokens-1) {
        tokens[pos+1] = sample(logits, temp, topp, topk);
        if((151643 == tokens[pos+1]) || (151645 == tokens[pos+1])) {
          break;
        }
        std::cout << tokenizer->decode({tokens[pos+1]}) << std::flush;
      }
    }

    gettimeofday(&tve, NULL);
    printf("(%ld tokens/s)\n", pos*1000000/((tve.tv_sec*1000000+tve.tv_usec)-(tvs.tv_sec*1000000+tvs.tv_usec)));

    free_transformer(transformer);

    exit(0);
}
