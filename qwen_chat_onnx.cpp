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

// ----------------------------------------------------------------------------
// Transformer model
typedef struct {
    int dim; // transformer dimension
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int batch_size;
} Config;

typedef struct {
    vector<int64_t> input_ids;
    vector<int64_t> attention_mask;
    vector<int64_t> position_ids;
    int past_len;
    vector<vector<float>> past_key_values_key;
    vector<vector<float>> past_key_values_value;

    vector<float> logits;
    int present_len;
} RunState;

typedef struct {
    Ort::Session session { nullptr };
    Ort::MemoryInfo allocator_info { nullptr };
    Config config;
    RunState state;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    s->input_ids = vector<int64_t>(p->batch_size * 1);
    s->attention_mask = vector<int64_t>(p->batch_size * 1);
    s->position_ids = vector<int64_t>(p->batch_size * 1);
    s->past_len = 0;
    s->past_key_values_key = vector<vector<float>>(p->n_layers);
    s->past_key_values_value = vector<vector<float>>(p->n_layers);
    for(int i=0; i<p->n_layers; i++) {
        s->past_key_values_key[i] = vector<float>(p->batch_size * p->n_heads * 1 * p->dim);
        s->past_key_values_value[i] = vector<float>(p->batch_size * p->n_heads * 1 * p->dim);
    }

    s->logits = vector<float>(p->batch_size * 1 * p->vocab_size);
    s->present_len = 0;
}

void free_run_state(RunState* s, Config* p) {
}

void build_transformer(Transformer *t, const char* onnx_file_path) {
    t->config.dim = 64;
    t->config.n_layers = 24;
    t->config.n_heads = 2;
    t->config.vocab_size = 151936;
    t->config.seq_len = 32000;
    t->config.batch_size = 1;

    // ONNX Transformer
    Ort::SessionOptions session_option;
    //session_option.SetIntraOpNumThreads(6);
    //session_option.SetInterOpNumThreads(1);
    static Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "qwen_chat_onnx" };
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
    session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    t->allocator_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    t->session = Ort::Session(env, onnx_file_path, session_option);

    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    free_run_state(&t->state, &t->config);
}

void forward_transformer(Transformer* t, vector<int> tokens) {
    vector<const char*> transformer_input_names { "input_ids", "attention_mask", "position_ids",
        "past_key_values_0_key", "past_key_values_0_value", "past_key_values_1_key", "past_key_values_1_value",
        "past_key_values_2_key", "past_key_values_2_value", "past_key_values_3_key", "past_key_values_3_value",
        "past_key_values_4_key", "past_key_values_4_value", "past_key_values_5_key", "past_key_values_5_value",
        "past_key_values_6_key", "past_key_values_6_value", "past_key_values_7_key", "past_key_values_7_value",
        "past_key_values_8_key", "past_key_values_8_value", "past_key_values_9_key", "past_key_values_9_value",
        "past_key_values_10_key", "past_key_values_10_value", "past_key_values_11_key", "past_key_values_11_value",
        "past_key_values_12_key", "past_key_values_12_value", "past_key_values_13_key", "past_key_values_13_value",
        "past_key_values_14_key", "past_key_values_14_value", "past_key_values_15_key", "past_key_values_15_value",
        "past_key_values_16_key", "past_key_values_16_value", "past_key_values_17_key", "past_key_values_17_value",
        "past_key_values_18_key", "past_key_values_18_value", "past_key_values_19_key", "past_key_values_19_value",
        "past_key_values_20_key", "past_key_values_20_value", "past_key_values_21_key", "past_key_values_21_value",
        "past_key_values_22_key", "past_key_values_22_value", "past_key_values_23_key", "past_key_values_23_value", };
    vector<const char*> transformer_output_names { "logits",
        "present_0_key", "present_0_value", "present_1_key", "present_1_value",
        "present_2_key", "present_2_value", "present_3_key", "present_3_value",
        "present_4_key", "present_4_value", "present_5_key", "present_5_value",
        "present_6_key", "present_6_value", "present_7_key", "present_7_value",
        "present_8_key", "present_8_value", "present_9_key", "present_9_value",
        "present_10_key", "present_10_value", "present_11_key", "present_11_value",
        "present_12_key", "present_12_value", "present_13_key", "present_13_value",
        "present_14_key", "present_14_value", "present_15_key", "present_15_value",
        "present_16_key", "present_16_value", "present_17_key", "present_17_value",
        "present_18_key", "present_18_value", "present_19_key", "present_19_value",
        "present_20_key", "present_20_value", "present_21_key", "present_21_value",
        "present_22_key", "present_22_value", "present_23_key", "present_23_value" };

    Config* c = &t->config;
    RunState* s = &t->state;

    int batch_size = c->batch_size;
    int in_len = tokens.size();
    int past_len = s->past_len;
    int out_len = past_len + in_len;

    std::array<int64_t, 2> input_ids_shape_{ batch_size, in_len };
    std::array<int64_t, 2> attention_mask_shape_{ batch_size, in_len };
    std::array<int64_t, 2> position_ids_shape_{ batch_size, in_len };
    std::array<int64_t, 4> past_key_values_shape_{ batch_size, c->n_heads, past_len, c->dim };

    s->input_ids.resize(in_len);
    s->attention_mask.resize(in_len);
    s->position_ids.resize(in_len);
    for(int i=0; i<in_len; i++) {
        s->input_ids[i] = tokens[i];
        s->attention_mask[i] = 1;
        s->position_ids[i] = past_len+i;
    }

    vector<Ort::Value> input_values;
    input_values.push_back(Ort::Value::CreateTensor<int64_t>(t->allocator_info, s->input_ids.data(), in_len, input_ids_shape_.data(), input_ids_shape_.size()));
    input_values.push_back(Ort::Value::CreateTensor<int64_t>(t->allocator_info, s->attention_mask.data(), in_len, attention_mask_shape_.data(), attention_mask_shape_.size()));
    input_values.push_back(Ort::Value::CreateTensor<int64_t>(t->allocator_info, s->position_ids.data(), in_len, position_ids_shape_.data(), position_ids_shape_.size()));
    for(int i=0; i<c->n_layers; i++) {
        input_values.push_back(Ort::Value::CreateTensor<float>(t->allocator_info, s->past_key_values_key[i].data(), c->n_heads * past_len * c->dim, past_key_values_shape_.data(), past_key_values_shape_.size()));
        input_values.push_back(Ort::Value::CreateTensor<float>(t->allocator_info, s->past_key_values_value[i].data(), c->n_heads * past_len * c->dim, past_key_values_shape_.data(), past_key_values_shape_.size()));
    }

    auto output_tensors = t->session.Run(Ort::RunOptions{ nullptr }, transformer_input_names.data(), input_values.data(), input_values.size(), transformer_output_names.data(), transformer_output_names.size());

    s->present_len = out_len;
    s->logits.resize(in_len * c->vocab_size);
    memcpy(s->logits.data(), output_tensors[0].GetTensorMutableData<float>(), in_len * c->vocab_size * sizeof(float));

    s->past_len = out_len;
    for(int i=0; i<c->n_layers; i++) {
        s->past_key_values_key[i].resize(c->n_heads * out_len * c->dim);
        s->past_key_values_value[i].resize(c->n_heads * out_len * c->dim);
        memcpy(s->past_key_values_key[i].data(), output_tensors[1+i*2].GetTensorMutableData<float>(), c->n_heads * out_len * c->dim * sizeof(float));
        memcpy(s->past_key_values_value[i].data(), output_tensors[2+i*2].GetTensorMutableData<float>(), c->n_heads * out_len * c->dim * sizeof(float));
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

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

// ./inference MODEL PROMPT
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
    Transformer transformer;
    build_transformer(&transformer, model_path.c_str());

    Config* c = &transformer.config;

    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    auto tokens = tokenizer->encode(prompt, prompt.length());
    int pos = tokens.size();

    while(1) {
      std::cout << "User: " << std::flush;
      string user_input;
      getline(cin, user_input);

      string format_input = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";

      // tokenize input
      auto tokens_input = tokenizer->encode(format_input, format_input.length());
      for(int i=0; i<tokens_input.size(); i++) {
        tokens.push_back(tokens_input[i]);
      }
      forward_transformer(&transformer, tokens);
      pos += tokens_input.size();

      std::cout << "AI: " << std::flush;
      int output_length = 0;

      struct timeval tvs, tve;
      gettimeofday(&tvs, NULL);

      int out_count = 0;

      // feed forward
      vector<float> logits(c->vocab_size, 0);
      for (; pos < c->seq_len; pos++) {
        memcpy(logits.data(), &transformer.state.logits.data()[(tokens.size()-1)*c->vocab_size], c->vocab_size);
        int next = sample(logits, temp, topp, topk);
        tokens.clear();
        tokens.push_back(next);

        out_count++;

        output_length++;
        std::cout << tokenizer->decode({next}) << std::flush;
        if((151643 == next) || (151645 == next)) {
          std::cout << std::endl;
          pos++;

          gettimeofday(&tve, NULL);
          printf("(%ld tokens/s)\n", out_count*1000000/((tve.tv_sec*1000000+tve.tv_usec)-(tvs.tv_sec*1000000+tvs.tv_usec)));

          break;
        }
        forward_transformer(&transformer, tokens);
      }
      tokens.clear();
    }

    free_transformer(&transformer);

    exit(0);
}
