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

int build_transformer(const char* checkpoint, void** p);
void run_transformer(int token, float* embedding, int pos, float* logits, void* p);
void free_transformer(void* p);

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


// Whisper log mel spectrogram
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_N_FFT_HALF  (WHISPER_N_FFT / 2 + 1)
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30
#define WHISPER_N_SAMPLES   (WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE)

#define SIN_COS_N_COUNT WHISPER_N_FFT

struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

    whisper_global_cache() {
        fill_sin_cos_table();
        fill_hann_window(sizeof(hann_window)/sizeof(hann_window[0]), true, hann_window);
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float * output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
} global_cache;

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*global_cache.cos_vals[idx]; // cos(t)
            im -= in[n]*global_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx]; // cos(t)
        float im = -global_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct whisper_mel_data {
    int n_len;
    int n_len_org;
    int n_mel;
    float * data;
};

void log_mel_spectrogram(const float * hann, const std::vector<float> & samples, int n_samples, const whisper_filters & filters, whisper_mel_data & mel) {
    const auto frame_size = WHISPER_N_FFT;
    const auto frame_step = WHISPER_HOP_LENGTH;
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);
    int n_fft = filters.n_fft;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (int i=0; i < std::min(n_samples / frame_step + 1, mel.n_len); i++) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;

            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }

            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }

            sum = log10(std::max(sum, 1e-10));

            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (int i=0; i < mel.n_len; i++) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

void calculate_mel(vector<float> ssamples, std::vector<float>& host_mel_data, whisper_mel_data& mel) {
    // Read whisper filter data
    whisper_filters filters;
    filters.n_mel = 80;
    filters.n_fft = 1 + WHISPER_N_FFT / 2;
    filters.data.resize(filters.n_mel * filters.n_fft);
    FILE* fp = fopen("tiny_filter.bin", "rb");
    fread(filters.data.data(), sizeof(float), filters.data.size(), fp);
    fclose(fp);

    // Hann window
    const float * hann = global_cache.hann_window;

    // Calculate the length of padding
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = WHISPER_N_FFT / 2;

    const int n_samples = int(ssamples.size());
    const float * samples = ssamples.data();

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel     = filters.n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - WHISPER_N_FFT) / WHISPER_HOP_LENGTH;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - WHISPER_N_FFT) / WHISPER_HOP_LENGTH;

    host_mel_data.resize(mel.n_len * mel.n_mel);
    mel.data = host_mel_data.data();

    log_mel_spectrogram(hann, samples_padded, n_samples + stage_2_pad, filters, mel);

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    for(int i=0; i<100; i++) {
      printf("%.5f    ", mel.data[i]);
    }
    printf("\n");
}

struct wave_header
{
    uint8_t  chunk_id[4];      //'RIFF'
    uint32_t chunk_size;
    uint8_t  format[4];        //'WAVE'
    uint8_t  subchunk1_id[4];  //'FMT'
    uint32_t subchunk1_size;   //PCM = 16
    uint16_t audio_format;     //PCM = 1
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;      //NumChannels * BitsPerSample / 8
    uint16_t bit_per_sample;
    uint8_t  subchunk2_id[4];  //'DATA'
    int32_t  subchunk2_size;
};

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL PROMPT TEMPERATURE"
                  << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string file_name = argv[2];
    std::string tokenizer_path = "qwen.tiktoken";
    temp = std::atof(argv[3]);

    // 读取wave文件
    FILE *fp = fopen(file_name.c_str(), "rb");
    wave_header header;

    if(!fp) {
      printf("Open wave file error\n");
      return 1;
    }

    fread(&header, sizeof(wave_header), 1, fp);

    vector<int16_t> waveFileData;

    int dataSize = header.subchunk2_size;
    printf("Wave file data size %d\n", dataSize);
    waveFileData.resize(dataSize/2);
    fread(waveFileData.data(), dataSize, 1, fp);

    fclose(fp);

    vector<float> waveData;
    for(int i=0; i<dataSize/2; i++) {
        waveData.push_back(static_cast<float>(waveFileData[i])/32768.0f);
    }

    std::vector<float> host_mel_data;
    whisper_mel_data mel;

    calculate_mel(waveData, host_mel_data, mel);

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
    //FILE *finput = fopen("mel_float.bin", "rb");
    memcpy(encoder_input_.data(), mel.data, 80 * 3000 * sizeof(float));
    //fclose(finput);

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

    int num_speech_tokens = 300;

    string transcribe_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nTranscribe the speech<|im_end|>\n<|im_start|>assistant\n";

    // tokenize prompt
    auto tokens_prompt = tokenizer->encode(transcribe_prompt, 32);
    for(int i=0; i<tokens_prompt.size(); i++) {
      tokens[num_speech_tokens+i] = tokens_prompt[i];
    }

    int pos = 0;
    vector<float> logits(vocab_size, 0);

    struct timeval tvs, tve;
    gettimeofday(&tvs, NULL);

    for (; pos < 512; pos++) {
      if(pos < num_speech_tokens) {
        run_transformer(-1, projector_output_.data(), pos, logits.data(), transformer);
      } else {
        run_transformer(tokens[pos], NULL, pos, logits.data(), transformer);
      }

      if(pos >= num_speech_tokens+tokens_prompt.size()-1) {
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
