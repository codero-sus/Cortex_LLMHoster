#include <fit.h>
#include <httplib.h>
#include <llama.h>
#include <nlohmann/json.hpp>

#include "prompt_cache.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <functional>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
constexpr std::size_t kMaxRequestBytes = 16 * 1024 * 1024;
constexpr std::uint32_t kDefaultMaxTokens = 256;
constexpr std::size_t kMaxStopSequences = 16;
constexpr std::size_t kMaxStopLength = 256;

class ApiError final : public std::runtime_error {
 public:
  ApiError(int status, std::string code, std::string message)
      : std::runtime_error(std::move(message)), status_(status), code_(std::move(code)) {}

  int status() const noexcept { return status_; }
  const std::string& code() const noexcept { return code_; }

 private:
  int status_;
  std::string code_;
};

struct Options {
  std::string model_path;
  std::string model_alias;
  std::string api_base_path = "/v1";
  std::string health_path = "/health";
  std::string host = "127.0.0.1";
  std::uint32_t port = 0;
  std::uint32_t context_size = 2048;
  std::uint32_t batch_size = 256;
  std::uint32_t ubatch_size = 64;
  int threads = 0;
  int threads_batch = 0;
  int gpu_layers = 0;
  bool use_mmap = true;
  bool mlock = false;
  bool embedding = false;
  std::string flash_attention = "auto";
  std::string cache_type_k = "f16";
  std::string cache_type_v = "f16";
};

struct SamplingOptions {
  std::uint32_t max_tokens = kDefaultMaxTokens;
  std::uint32_t seed = LLAMA_DEFAULT_SEED;
  int top_k = 40;
  float temperature = 0.7F;
  float top_p = 0.95F;
  float frequency_penalty = 0.0F;
  float presence_penalty = 0.0F;
  std::vector<std::string> stop;
};

struct GenerationResult {
  std::string text;
  std::size_t prompt_tokens = 0;
  std::size_t completion_tokens = 0;
  std::string finish_reason = "length";
};

using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string trim_trailing_slashes(std::string value) {
  while (value.size() > 1 && value.back() == '/') {
    value.pop_back();
  }
  return value == "/" ? "" : value;
}

std::uint32_t parse_u32(const std::string& value, const std::string& name) {
  try {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("out of range");
    }
    return static_cast<std::uint32_t>(parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + name + ".");
  }
}

int parse_int(const std::string& value, const std::string& name) {
  try {
    std::size_t consumed = 0;
    const auto parsed = std::stol(value, &consumed, 10);
    if (consumed != value.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("out of range");
    }
    return static_cast<int>(parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + name + ".");
  }
}

void print_usage(const char* executable) {
  std::cout << "Usage: " << executable
            << " --model-path MODEL.gguf --model-alias NAME --port PORT [options]\n"
            << "  --host 127.0.0.1        Private listener (loopback only)\n"
            << "  --api-base-path /v1     OpenAI-compatible route prefix\n"
            << "  --health-path /health   Local health route\n"
            << "  --context-size N        Context tokens (default 2048)\n"
            << "  --batch-size N          Prompt batch size (default 256)\n"
            << "  --ubatch-size N         Physical micro-batch size (default 64)\n"
            << "  --threads N             Decode threads; 0 lets the engine choose\n"
            << "  --threads-batch N       Prompt-processing threads\n"
            << "  --gpu-layers N          GPU offload layers (-1 = all, -2 = memory-fit, 0 = CPU)\n"
            << "  --flash-attn auto|on|off\n"
            << "  --cache-type-k TYPE --cache-type-v TYPE\n"
            << "  --no-mmap --mlock --embedding\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  auto value_after = [&](int& index, const std::string& flag) -> std::string {
    if (index + 1 >= argc) {
      throw std::runtime_error("Missing value for " + flag + ".");
    }
    return argv[++index];
  };

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help" || flag == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (flag == "--model-path") {
      options.model_path = value_after(i, flag);
    } else if (flag == "--model-alias") {
      options.model_alias = value_after(i, flag);
    } else if (flag == "--host") {
      options.host = value_after(i, flag);
    } else if (flag == "--port") {
      options.port = parse_u32(value_after(i, flag), flag);
    } else if (flag == "--api-base-path") {
      options.api_base_path = value_after(i, flag);
    } else if (flag == "--health-path") {
      options.health_path = value_after(i, flag);
    } else if (flag == "--context-size") {
      options.context_size = parse_u32(value_after(i, flag), flag);
    } else if (flag == "--batch-size") {
      options.batch_size = parse_u32(value_after(i, flag), flag);
    } else if (flag == "--ubatch-size") {
      options.ubatch_size = parse_u32(value_after(i, flag), flag);
    } else if (flag == "--threads") {
      options.threads = parse_int(value_after(i, flag), flag);
    } else if (flag == "--threads-batch") {
      options.threads_batch = parse_int(value_after(i, flag), flag);
    } else if (flag == "--gpu-layers") {
      options.gpu_layers = parse_int(value_after(i, flag), flag);
    } else if (flag == "--flash-attn") {
      options.flash_attention = value_after(i, flag);
    } else if (flag == "--cache-type-k") {
      options.cache_type_k = value_after(i, flag);
    } else if (flag == "--cache-type-v") {
      options.cache_type_v = value_after(i, flag);
    } else if (flag == "--no-mmap") {
      options.use_mmap = false;
    } else if (flag == "--mlock") {
      options.mlock = true;
    } else if (flag == "--embedding") {
      options.embedding = true;
    } else {
      throw std::runtime_error("Unknown option: " + flag);
    }
  }

  if (options.model_path.empty() || options.model_alias.empty() || options.port == 0 ||
      options.port > 65535) {
    throw std::runtime_error("Model path, model alias, and a port from 1 to 65535 are required.");
  }
  if (options.host != "127.0.0.1") {
    throw std::runtime_error("Cortex's private GGUF process may only bind to 127.0.0.1.");
  }
  if (options.context_size < 128 || options.batch_size == 0 || options.ubatch_size == 0 ||
      options.threads < 0 || options.threads_batch < 0 || options.gpu_layers < -2) {
    throw std::runtime_error("Context, batch, thread, or GPU-layer settings are out of range.");
  }
  options.batch_size = std::min(options.batch_size, options.context_size);
  options.ubatch_size = std::min(options.ubatch_size, options.batch_size);
  if (options.flash_attention != "auto" && options.flash_attention != "on" &&
      options.flash_attention != "off") {
    throw std::runtime_error("--flash-attn must be auto, on, or off.");
  }
  options.api_base_path = trim_trailing_slashes(options.api_base_path);
  if ((!options.api_base_path.empty() && options.api_base_path.front() != '/') ||
      options.health_path.empty() || options.health_path.front() != '/' ||
      options.api_base_path.find_first_of("?#\\") != std::string::npos ||
      options.health_path.find_first_of("?#\\") != std::string::npos ||
      options.api_base_path.find("..") != std::string::npos ||
      options.health_path.find("..") != std::string::npos) {
    throw std::runtime_error("API and health paths must be absolute URL paths without dot segments.");
  }
  return options;
}

float json_float(const Json& object, const char* name, float fallback, float minimum,
                 float maximum) {
  if (!object.contains(name)) {
    return fallback;
  }
  const auto& value = object.at(name);
  if (!value.is_number()) {
    throw ApiError(400, "invalid_request", std::string(name) + " must be a number.");
  }
  const float parsed = value.get<float>();
  if (!std::isfinite(parsed) || parsed < minimum || parsed > maximum) {
    throw ApiError(400, "invalid_request", std::string(name) + " is outside its allowed range.");
  }
  return parsed;
}

std::uint32_t json_max_tokens(const Json& object) {
  const char* key = object.contains("max_completion_tokens") ? "max_completion_tokens" : "max_tokens";
  if (!object.contains(key)) {
    return kDefaultMaxTokens;
  }
  const auto& value = object.at(key);
  if (!value.is_number_integer()) {
    throw ApiError(400, "invalid_request", std::string(key) + " must be a positive integer.");
  }
  const auto parsed = value.get<std::int64_t>();
  if (parsed < 1 || parsed > 131072) {
    throw ApiError(400, "invalid_request", std::string(key) + " must be between 1 and 131072.");
  }
  return static_cast<std::uint32_t>(parsed);
}

SamplingOptions parse_sampling(const Json& payload) {
  SamplingOptions options;
  options.max_tokens = json_max_tokens(payload);
  options.temperature = json_float(payload, "temperature", 0.7F, 0.0F, 2.0F);
  options.top_p = json_float(payload, "top_p", 0.95F, 0.01F, 1.0F);
  options.frequency_penalty = json_float(payload, "frequency_penalty", 0.0F, -2.0F, 2.0F);
  options.presence_penalty = json_float(payload, "presence_penalty", 0.0F, -2.0F, 2.0F);
  if (payload.contains("top_k")) {
    const auto& value = payload.at("top_k");
    if (!value.is_number_integer() || value.get<std::int64_t>() < 0 ||
        value.get<std::int64_t>() > 1000) {
      throw ApiError(400, "invalid_request", "top_k must be an integer between 0 and 1000.");
    }
    options.top_k = value.get<int>();
  }
  if (payload.contains("seed")) {
    const auto& value = payload.at("seed");
    if (!value.is_number_integer() || value.get<std::int64_t>() < 0 ||
        value.get<std::int64_t>() > std::numeric_limits<std::uint32_t>::max()) {
      throw ApiError(400, "invalid_request", "seed must be a non-negative 32-bit integer.");
    }
    options.seed = value.get<std::uint32_t>();
  }
  if (payload.contains("stop")) {
    const auto& stop = payload.at("stop");
    if (stop.is_string()) {
      options.stop.push_back(stop.get<std::string>());
    } else if (stop.is_array()) {
      if (stop.size() > kMaxStopSequences) {
        throw ApiError(400, "invalid_request", "Too many stop sequences.");
      }
      for (const auto& item : stop) {
        if (!item.is_string()) {
          throw ApiError(400, "invalid_request", "Stop sequences must be strings.");
        }
        options.stop.push_back(item.get<std::string>());
      }
    } else {
      throw ApiError(400, "invalid_request", "stop must be a string or an array of strings.");
    }
    if (std::any_of(options.stop.begin(), options.stop.end(), [](const std::string& item) {
          return item.empty() || item.size() > kMaxStopLength;
        })) {
      throw ApiError(400, "invalid_request", "Stop sequences must contain 1 to 256 bytes.");
    }
  }
  return options;
}

std::string get_text(const Json& value, const char* name) {
  if (!value.is_string()) {
    throw ApiError(400, "invalid_request", std::string(name) + " must be a string.");
  }
  auto text = value.get<std::string>();
  if (text.find('\0') != std::string::npos) {
    throw ApiError(400, "invalid_request", std::string(name) + " may not contain NUL bytes.");
  }
  return text;
}

std::string make_chat_prompt(const Json& payload, const llama_model* model) {
  if (!payload.contains("messages") || !payload.at("messages").is_array() ||
      payload.at("messages").empty()) {
    throw ApiError(400, "invalid_request", "messages must be a non-empty array.");
  }
  std::vector<std::string> roles;
  std::vector<std::string> contents;
  roles.reserve(payload.at("messages").size());
  contents.reserve(payload.at("messages").size());
  for (const auto& item : payload.at("messages")) {
    if (!item.is_object() || !item.contains("role") || !item.contains("content")) {
      throw ApiError(400, "invalid_request", "Each message needs a role and text content.");
    }
    roles.push_back(get_text(item.at("role"), "message role"));
    contents.push_back(get_text(item.at("content"), "message content"));
  }

  const char* chat_template = llama_model_chat_template(model, nullptr);
  if (chat_template == nullptr) {
    throw ApiError(422, "unsupported_model_capability",
                   "This GGUF has no supported chat template; use the completions endpoint.");
  }
  std::vector<llama_chat_message> messages;
  messages.reserve(roles.size());
  for (std::size_t i = 0; i < roles.size(); ++i) {
    messages.push_back({roles[i].c_str(), contents[i].c_str()});
  }
  const auto required = llama_chat_apply_template(chat_template, messages.data(), messages.size(),
                                                   true, nullptr, 0);
  if (required < 0) {
    throw ApiError(422, "unsupported_model_capability",
                   "The model's chat template is not supported by this llama.cpp build.");
  }
  std::vector<char> buffer(static_cast<std::size_t>(required) + 1, '\0');
  const auto written = llama_chat_apply_template(chat_template, messages.data(), messages.size(),
                                                  true, buffer.data(),
                                                  static_cast<std::int32_t>(buffer.size()));
  if (written < 0 || written > required) {
    throw ApiError(422, "unsupported_model_capability", "Could not apply the GGUF chat template.");
  }
  return std::string(buffer.data(), static_cast<std::size_t>(written));
}

std::size_t complete_utf8_prefix(std::string_view text) {
  std::size_t index = 0;
  while (index < text.size()) {
    const auto lead = static_cast<unsigned char>(text[index]);
    std::size_t width = 1;
    if ((lead & 0xe0U) == 0xc0U) {
      width = 2;
    } else if ((lead & 0xf0U) == 0xe0U) {
      width = 3;
    } else if ((lead & 0xf8U) == 0xf0U) {
      width = 4;
    }
    if (index + width > text.size()) {
      return index;
    }
    if (width > 1) {
      for (std::size_t offset = 1; offset < width; ++offset) {
        if ((static_cast<unsigned char>(text[index + offset]) & 0xc0U) != 0x80U) {
          width = 1;
          break;
        }
      }
    }
    index += width;
  }
  return index;
}

class StopFilter {
 public:
  explicit StopFilter(const std::vector<std::string>& stop) : stop_(stop) {
    for (const auto& item : stop_) {
      hold_bytes_ = std::max(hold_bytes_, item.size() - 1);
    }
  }

  template <typename Emit>
  bool push(std::string_view piece, Emit&& emit) {
    pending_.append(piece.data(), piece.size());
    std::size_t first = std::string::npos;
    for (const auto& sequence : stop_) {
      const auto match = pending_.find(sequence);
      if (match != std::string::npos && (first == std::string::npos || match < first)) {
        first = match;
      }
    }
    if (first != std::string::npos) {
      const bool written = emit_ready(first, emit);
      pending_.clear();
      stopped_ = true;
      return written;
    }
    const auto safe_bytes = pending_.size() > hold_bytes_ ? pending_.size() - hold_bytes_ : 0;
    return emit_ready(safe_bytes, emit);
  }

  template <typename Emit>
  bool finish(Emit&& emit) {
    if (stopped_ || pending_.empty()) {
      return true;
    }
    return emit_ready(pending_.size(), emit);
  }

  bool stopped() const noexcept { return stopped_; }

 private:
  template <typename Emit>
  bool emit_ready(std::size_t requested, Emit&& emit) {
    const auto safe = complete_utf8_prefix(std::string_view(pending_).substr(0, requested));
    if (safe == 0) {
      return true;
    }
    const auto fragment = pending_.substr(0, safe);
    pending_.erase(0, safe);
    return emit(fragment);
  }

  std::vector<std::string> stop_;
  std::string pending_;
  std::size_t hold_bytes_ = 0;
  bool stopped_ = false;
};

void set_batch_tokens(llama_batch& batch, const std::vector<llama_token>& tokens,
                      std::size_t offset, std::size_t count, bool output_all) {
  batch.n_tokens = static_cast<std::int32_t>(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto index = static_cast<std::int32_t>(i);
    batch.token[index] = tokens[offset + i];
    batch.pos[index] = static_cast<llama_pos>(offset + i);
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = 0;
    batch.logits[index] = (output_all || i + 1 == count) ? 1 : 0;
  }
}

void set_single_token(llama_batch& batch, llama_token token, std::size_t position) {
  batch.n_tokens = 1;
  batch.token[0] = token;
  batch.pos[0] = static_cast<llama_pos>(position);
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = 0;
  batch.logits[0] = 1;
}

struct BatchGuard {
  explicit BatchGuard(std::int32_t size) : batch(llama_batch_init(size, 0, 1)) {
    if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr || batch.logits == nullptr) {
      llama_batch_free(batch);
      throw std::runtime_error("Could not allocate an inference batch.");
    }
  }
  ~BatchGuard() { llama_batch_free(batch); }
  BatchGuard(const BatchGuard&) = delete;
  BatchGuard& operator=(const BatchGuard&) = delete;
  llama_batch batch;
};

struct RuntimeModel {
  ModelPtr model{nullptr, llama_model_free};
  ContextPtr context{nullptr, llama_free};
  const llama_vocab* vocab = nullptr;
  int embedding_size = 0;
  bool embeddings_enabled = false;
  std::string alias;
  std::uint32_t context_size = 0;
  std::uint32_t batch_size = 0;
  std::mutex inference_mutex;
};

class Engine {
 public:
  explicit Engine(const Options& options) {
    if (options.model_path.size() < 5 || lower(options.model_path.substr(options.model_path.size() - 5)) != ".gguf") {
      throw std::runtime_error("The native backend requires an existing .gguf model file.");
    }
    llama_backend_init();
    backend_initialized_ = true;

    auto model_params = llama_model_default_params();
    if (options.gpu_layers != -2) {
      model_params.n_gpu_layers = options.gpu_layers;
    }
    if (options.use_mmap && options.mlock) {
      model_params.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
    } else if (options.use_mmap) {
      model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    } else if (options.mlock) {
      model_params.load_mode = LLAMA_LOAD_MODE_MLOCK;
    } else {
      model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = options.context_size;
    context_params.n_batch = options.batch_size;
    context_params.n_ubatch = options.ubatch_size;
    context_params.n_seq_max = 1;
    context_params.n_threads = options.threads;
    context_params.n_threads_batch = options.threads_batch;
    context_params.embeddings = options.embedding;
    context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    context_params.flash_attn_type = options.flash_attention == "on"
                                         ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                         : options.flash_attention == "off"
                                               ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                               : LLAMA_FLASH_ATTN_TYPE_AUTO;
    context_params.type_k = parse_cache_type(options.cache_type_k);
    context_params.type_v = parse_cache_type(options.cache_type_v);

    std::vector<float> tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    std::vector<std::size_t> memory_margins;
    if (options.gpu_layers == -2) {
      tensor_split.assign(llama_max_devices(), 0.0F);
      tensor_buft_overrides.resize(llama_max_tensor_buft_overrides());
      memory_margins.assign(llama_max_devices(), std::size_t{1024} * 1024 * 1024);
      const auto fit_status = common_fit_params(
          options.model_path.c_str(), &model_params, &context_params, tensor_split.data(),
          tensor_buft_overrides.data(), memory_margins.data(),
          std::min<std::uint32_t>(options.context_size, 1024), nullptr, GGML_LOG_LEVEL_WARN);
      if (fit_status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
        throw std::runtime_error(
            "Automatic GPU memory fitting failed. Try a smaller context or a quantized GGUF.");
      }
      std::cerr << "Cortex memory-fit selected ";
      if (model_params.n_gpu_layers < 0) {
        std::cerr << "all compatible GPU layers";
      } else if (model_params.n_gpu_layers == 0) {
        std::cerr << "CPU-only placement";
      } else {
        std::cerr << model_params.n_gpu_layers << " GPU layers";
      }
      std::cerr << " at context " << context_params.n_ctx << ".\n";
    }

    runtime_.model.reset(llama_model_load_from_file(options.model_path.c_str(), model_params));
    if (!runtime_.model) {
      throw std::runtime_error("The GGUF model could not be loaded by the native inference core.");
    }
    runtime_.vocab = llama_model_get_vocab(runtime_.model.get());
    runtime_.alias = options.model_alias;
    runtime_.embeddings_enabled = options.embedding;
    runtime_.embedding_size = llama_model_n_embd_out(runtime_.model.get());
    runtime_.context_size = options.context_size;
    runtime_.batch_size = options.batch_size;

    runtime_.context.reset(llama_init_from_model(runtime_.model.get(), context_params));
    if (!runtime_.context) {
      throw std::runtime_error("The native inference context could not be created.");
    }
    runtime_.context_size = llama_n_ctx(runtime_.context.get());
    runtime_.batch_size = llama_n_batch(runtime_.context.get());
  }

  ~Engine() {
    runtime_.context.reset();
    runtime_.model.reset();
    if (backend_initialized_) {
      llama_backend_free();
    }
  }

  std::vector<llama_token> tokenize(const std::string& prompt) const {
    if (prompt.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      throw ApiError(413, "request_too_large", "The prompt is too large to tokenize.");
    }
    const auto text_length = static_cast<std::int32_t>(prompt.size());
    auto needed = llama_tokenize(runtime_.vocab, prompt.data(), text_length, nullptr, 0, true, true);
    if (needed == std::numeric_limits<std::int32_t>::min()) {
      throw ApiError(400, "invalid_request", "The prompt could not be tokenized.");
    }
    if (needed < 0) {
      needed = -needed;
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(std::max(needed, 1)));
    const auto actual = llama_tokenize(runtime_.vocab, prompt.data(), text_length, tokens.data(),
                                       static_cast<std::int32_t>(tokens.size()), true, true);
    if (actual < 0) {
      throw ApiError(400, "invalid_request", "The prompt could not be tokenized.");
    }
    tokens.resize(static_cast<std::size_t>(actual));
    if (tokens.empty()) {
      throw ApiError(400, "invalid_request", "The prompt produced no model tokens.");
    }
    if (tokens.size() > runtime_.context_size) {
      throw ApiError(400, "context_length_exceeded", "The prompt exceeds the configured context size.");
    }
    return tokens;
  }

  GenerationResult generate(const std::vector<llama_token>& prompt_tokens,
                           const SamplingOptions& sampling,
                           const std::function<bool(std::string_view)>& emit) {
    std::lock_guard<std::mutex> guard(runtime_.inference_mutex);
    if (prompt_tokens.empty() || prompt_tokens.size() > runtime_.context_size) {
      throw ApiError(400, "context_length_exceeded", "The prompt exceeds the configured context size.");
    }
    const auto prompt_eval_start = prepare_prompt_cache(prompt_tokens);
    try {
      evaluate_prompt(prompt_tokens, false, prompt_eval_start);
    } catch (...) {
      llama_memory_clear(llama_get_memory(runtime_.context.get()), true);
      cached_tokens_.clear();
      throw;
    }
    cached_tokens_.assign(prompt_tokens.begin(), prompt_tokens.end());

    auto chain_params = llama_sampler_chain_default_params();
    SamplerPtr sampler(llama_sampler_chain_init(chain_params), llama_sampler_free);
    if (!sampler) {
      throw std::runtime_error("Could not initialize the sampler chain.");
    }
    if (sampling.frequency_penalty != 0.0F || sampling.presence_penalty != 0.0F) {
      llama_sampler_chain_add(
          sampler.get(),
          llama_sampler_init_penalties(llama_vocab_n_tokens(runtime_.vocab), 64, 1.0F,
                                       sampling.frequency_penalty, sampling.presence_penalty));
    }
    if (sampling.top_k > 0) {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(sampling.top_k));
    }
    if (sampling.top_p < 1.0F) {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(sampling.top_p, 1));
    }
    if (sampling.temperature <= 0.0F) {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    } else {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(sampling.temperature));
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(sampling.seed));
    }
    for (const auto token : prompt_tokens) {
      llama_sampler_accept(sampler.get(), token);
    }

    GenerationResult result;
    result.prompt_tokens = prompt_tokens.size();
    const auto available = runtime_.context_size - prompt_tokens.size() + 1;
    const auto token_limit = std::min<std::uint32_t>(sampling.max_tokens, available);
    std::size_t next_position = prompt_tokens.size();
    StopFilter filter(sampling.stop);
    bool client_disconnected = false;
    auto filtered_emit = [&](std::string_view text) {
      if (text.empty()) {
        return true;
      }
      result.text.append(text.data(), text.size());
      if (emit && !emit(text)) {
        client_disconnected = true;
        return false;
      }
      return true;
    };

    // Reuse the one-token decode batch across generated tokens instead of
    // allocating and freeing llama_batch storage for every decode step.
    BatchGuard decode_batch(1);
    for (std::uint32_t index = 0; index < token_limit; ++index) {
      const auto token = llama_sampler_sample(sampler.get(), runtime_.context.get(), -1);
      if (llama_vocab_is_eog(runtime_.vocab, token)) {
        result.finish_reason = "stop";
        break;
      }
      const auto piece = token_piece(token);
      ++result.completion_tokens;
      if (!filter.push(piece, filtered_emit)) {
        result.finish_reason = filter.stopped() ? "stop" : "cancelled";
        break;
      }
      if (filter.stopped()) {
        result.finish_reason = "stop";
        break;
      }
      if (index + 1 == token_limit || next_position >= runtime_.context_size) {
        result.finish_reason = "length";
        break;
      }

      set_single_token(decode_batch.batch, token, next_position);
      if (llama_decode(runtime_.context.get(), decode_batch.batch) != 0) {
        throw ApiError(500, "native_inference_failed", "The native model could not decode the next token.");
      }
      cached_tokens_.push_back(token);
      ++next_position;
    }
    if (!client_disconnected && !filter.stopped() && !filter.finish(filtered_emit)) {
      result.finish_reason = "cancelled";
    }
    return result;
  }

  std::vector<float> embed(const std::vector<llama_token>& tokens) {
    if (!runtime_.embeddings_enabled) {
      throw ApiError(422, "unsupported_model_capability", "Embeddings are disabled for this model.");
    }
    if (runtime_.embedding_size <= 0) {
      throw ApiError(501, "backend_operation_unavailable", "The model has no embedding output.");
    }
    std::lock_guard<std::mutex> guard(runtime_.inference_mutex);
    if (tokens.empty() || tokens.size() > runtime_.context_size) {
      throw ApiError(400, "context_length_exceeded", "The input exceeds the configured context size.");
    }
    llama_memory_clear(llama_get_memory(runtime_.context.get()), true);
    cached_tokens_.clear();

    std::vector<float> pooled(static_cast<std::size_t>(runtime_.embedding_size), 0.0F);
    std::size_t count = 0;
    std::size_t offset = 0;
    BatchGuard batch(static_cast<std::int32_t>(runtime_.batch_size));
    while (offset < tokens.size()) {
      const auto batch_count = std::min<std::size_t>(runtime_.batch_size, tokens.size() - offset);
      set_batch_tokens(batch.batch, tokens, offset, batch_count, true);
      if (llama_decode(runtime_.context.get(), batch.batch) != 0) {
        throw ApiError(500, "native_inference_failed", "The native model could not compute embeddings.");
      }
      for (std::size_t i = 0; i < batch_count; ++i) {
        const auto* values = llama_get_embeddings_ith(runtime_.context.get(), static_cast<std::int32_t>(i));
        if (values == nullptr) {
          throw ApiError(501, "backend_operation_unavailable", "The model did not return token embeddings.");
        }
        for (int dimension = 0; dimension < runtime_.embedding_size; ++dimension) {
          pooled[static_cast<std::size_t>(dimension)] += values[dimension];
        }
        ++count;
      }
      offset += batch_count;
    }
    if (count == 0) {
      throw ApiError(400, "invalid_request", "The input produced no embedding tokens.");
    }
    double squared_norm = 0.0;
    for (auto& value : pooled) {
      value /= static_cast<float>(count);
      squared_norm += static_cast<double>(value) * value;
    }
    const auto norm = std::sqrt(squared_norm);
    if (norm > 0.0 && std::isfinite(norm)) {
      for (auto& value : pooled) {
        value = static_cast<float>(value / norm);
      }
    }
    return pooled;
  }

  const std::string& alias() const noexcept { return runtime_.alias; }
  const llama_model* model() const noexcept { return runtime_.model.get(); }

 private:
  std::size_t prepare_prompt_cache(const std::vector<llama_token>& prompt_tokens) {
    const auto plan = cortex::native::plan_prompt_cache(cached_tokens_, prompt_tokens);
    auto* memory = llama_get_memory(runtime_.context.get());
    auto keep = plan.keep_tokens;
    if (plan.clear_memory ||
        (keep < cached_tokens_.size() &&
         !llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(keep), -1))) {
      llama_memory_clear(memory, true);
      keep = 0;
    }
    cached_tokens_.resize(keep);
    return keep;
  }

  static ggml_type parse_cache_type(const std::string& value) {
    const auto type = lower(value);
    if (type == "f32") return GGML_TYPE_F32;
    if (type == "f16") return GGML_TYPE_F16;
    if (type == "bf16") return GGML_TYPE_BF16;
    if (type == "q8_0") return GGML_TYPE_Q8_0;
    if (type == "q5_0") return GGML_TYPE_Q5_0;
    if (type == "q5_1") return GGML_TYPE_Q5_1;
    if (type == "q4_0") return GGML_TYPE_Q4_0;
    if (type == "q4_1") return GGML_TYPE_Q4_1;
    throw std::runtime_error("Unsupported KV-cache type: " + value);
  }

  std::string token_piece(llama_token token) const {
    std::array<char, 256> small_buffer{};
    auto size = llama_token_to_piece(runtime_.vocab, token, small_buffer.data(),
                                     static_cast<std::int32_t>(small_buffer.size()), 0, true);
    if (size >= 0) {
      return std::string(small_buffer.data(), static_cast<std::size_t>(size));
    }
    if (size == std::numeric_limits<std::int32_t>::min()) {
      throw ApiError(500, "native_inference_failed", "The native model emitted an invalid token.");
    }
    std::vector<char> large_buffer(static_cast<std::size_t>(-size));
    size = llama_token_to_piece(runtime_.vocab, token, large_buffer.data(),
                                static_cast<std::int32_t>(large_buffer.size()), 0, true);
    if (size < 0) {
      throw ApiError(500, "native_inference_failed", "The native model emitted an invalid token.");
    }
    return std::string(large_buffer.data(), static_cast<std::size_t>(size));
  }

  void evaluate_prompt(const std::vector<llama_token>& tokens, bool output_all,
                       std::size_t offset) {
    BatchGuard batch(static_cast<std::int32_t>(runtime_.batch_size));
    while (offset < tokens.size()) {
      const auto batch_count = std::min<std::size_t>(runtime_.batch_size, tokens.size() - offset);
      set_batch_tokens(batch.batch, tokens, offset, batch_count, output_all);
      if (llama_decode(runtime_.context.get(), batch.batch) != 0) {
        throw ApiError(400, "context_length_exceeded", "The native model could not evaluate this prompt.");
      }
      offset += batch_count;
    }
  }

  RuntimeModel runtime_;
  std::vector<llama_token> cached_tokens_;
  bool backend_initialized_ = false;
};

Json error_body(const std::string& message, const std::string& code,
                const std::string& type = "invalid_request_error") {
  return {{"error", {{"message", message}, {"type", type}, {"code", code}}}};
}

void set_error(httplib::Response& response, int status, const std::string& message,
               const std::string& code, const std::string& type = "invalid_request_error") {
  response.status = status;
  response.set_content(error_body(message, code, type).dump(-1, ' ', false,
                                                        Json::error_handler_t::replace),
                       "application/json");
}

Json parse_body(const httplib::Request& request) {
  try {
    auto payload = Json::parse(request.body);
    if (!payload.is_object()) {
      throw ApiError(400, "invalid_request", "JSON request body must be an object.");
    }
    return payload;
  } catch (const Json::parse_error&) {
    throw ApiError(400, "invalid_request", "Request body must be valid JSON.");
  }
}

bool include_stream_usage(const Json& payload) {
  if (!payload.contains("stream_options")) {
    return false;
  }
  const auto& stream_options = payload.at("stream_options");
  if (!stream_options.is_object()) {
    throw ApiError(400, "invalid_request", "stream_options must be an object.");
  }
  if (!stream_options.contains("include_usage")) {
    return false;
  }
  if (!stream_options.at("include_usage").is_boolean()) {
    throw ApiError(400, "invalid_request", "stream_options.include_usage must be a boolean.");
  }
  return stream_options.at("include_usage").get<bool>();
}

std::string sse_data(const Json& event) {
  return "data: " + event.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n";
}

std::string make_id(const char* prefix) {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::string(prefix) + std::to_string(
                                    std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

std::uint64_t unix_time_seconds() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count());
}

void register_routes(httplib::Server& server, Engine& engine, const Options& options,
                     const llama_model* model) {
  const auto base = options.api_base_path;
  server.Get(options.health_path, [](const httplib::Request&, httplib::Response& response) {
    response.set_header("Cache-Control", "no-store");
    response.set_content(R"({"status":"ok"})", "application/json");
  });

  auto inference_handler = [&](bool chat) {
    return [&engine, &options, model, chat](const httplib::Request& request,
                                            httplib::Response& response) {
      try {
        const auto payload = parse_body(request);
        bool streaming = false;
        if (payload.contains("stream")) {
          if (!payload.at("stream").is_boolean()) {
            throw ApiError(400, "invalid_request", "The stream field must be a boolean.");
          }
          streaming = payload.at("stream").get<bool>();
        }
        const bool stream_usage = streaming && include_stream_usage(payload);
        const auto prompt = chat ? make_chat_prompt(payload, model)
                                 : get_text(payload.value("prompt", Json()), "prompt");
        const auto tokens = engine.tokenize(prompt);
        const auto sampling = parse_sampling(payload);
        if (streaming) {
          const auto id = make_id(chat ? "chatcmpl-" : "cmpl-");
          const auto created = unix_time_seconds();
          response.set_header("Cache-Control", "no-cache");
          response.set_header("X-Accel-Buffering", "no");
          response.set_chunked_content_provider(
              "text/event-stream",
              [&engine, &options, tokens, sampling, id, created, chat,
               stream_usage](std::size_t, httplib::DataSink& sink) {
                try {
                  const auto result = engine.generate(tokens, sampling, [&](std::string_view piece) {
                    if (!sink.is_writable()) {
                      return false;
                    }
                    Json delta = chat ? Json{{"content", std::string(piece)}}
                                      : Json{{"text", std::string(piece)}};
                    Json choice = {{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", nullptr}};
                    if (!chat) {
                      choice = {{"index", 0}, {"text", std::string(piece)}, {"finish_reason", nullptr}};
                    }
                    Json event = {{"id", id},
                                  {"object", chat ? "chat.completion.chunk" : "text_completion"},
                                  {"created", created},
                                  {"model", options.model_alias},
                                  {"choices", Json::array({choice})}};
                    const auto data = sse_data(event);
                    return sink.write(data.data(), data.size());
                  });
                  if (!sink.is_writable()) {
                    return false;
                  }
                  Json final_choice = {{"index", 0}, {"finish_reason", result.finish_reason}};
                  final_choice[chat ? "delta" : "text"] = chat ? Json::object() : Json("");
                  Json final_event = {{"id", id},
                                      {"object", chat ? "chat.completion.chunk" : "text_completion"},
                                      {"created", created},
                                      {"model", options.model_alias},
                                      {"choices", Json::array({final_choice})}};
                  auto final_data = sse_data(final_event);
                  if (stream_usage) {
                    Json usage_event = {{"id", id},
                                        {"object", chat ? "chat.completion.chunk" : "text_completion"},
                                        {"created", created},
                                        {"model", options.model_alias},
                                        {"choices", Json::array()},
                                        {"usage", {{"prompt_tokens", result.prompt_tokens},
                                                   {"completion_tokens", result.completion_tokens},
                                                   {"total_tokens", result.prompt_tokens +
                                                                        result.completion_tokens}}}};
                    final_data += sse_data(usage_event);
                  }
                  final_data += "data: [DONE]\n\n";
                  return sink.write(final_data.data(), final_data.size());
                } catch (const std::exception& error) {
                  std::cerr << "Cortex native stream failed: " << error.what() << '\n';
                  const auto failure = sse_data(error_body("The native model stream failed.",
                                                            "native_stream_failed", "server_error")) +
                                       "data: [DONE]\n\n";
                  return sink.write(failure.data(), failure.size());
                }
              });
          return;
        }

        const auto result = engine.generate(tokens, sampling, {});
        Json response_body;
        if (chat) {
          response_body = {{"id", make_id("chatcmpl-")},
                           {"object", "chat.completion"},
                           {"created", unix_time_seconds()},
                           {"model", options.model_alias},
                           {"choices", Json::array({{{"index", 0},
                                                     {"message", {{"role", "assistant"},
                                                                   {"content", result.text}}},
                                                     {"finish_reason", result.finish_reason}}})},
                           {"usage", {{"prompt_tokens", result.prompt_tokens},
                                      {"completion_tokens", result.completion_tokens},
                                      {"total_tokens", result.prompt_tokens + result.completion_tokens}}}};
        } else {
          response_body = {{"id", make_id("cmpl-")},
                           {"object", "text_completion"},
                           {"created", unix_time_seconds()},
                           {"model", options.model_alias},
                           {"choices", Json::array({{{"index", 0},
                                                     {"text", result.text},
                                                     {"finish_reason", result.finish_reason}}})},
                           {"usage", {{"prompt_tokens", result.prompt_tokens},
                                      {"completion_tokens", result.completion_tokens},
                                      {"total_tokens", result.prompt_tokens + result.completion_tokens}}}};
        }
        response.set_content(response_body.dump(-1, ' ', false, Json::error_handler_t::replace),
                             "application/json");
      } catch (const ApiError& error) {
        set_error(response, error.status(), error.what(), error.code());
      } catch (const std::exception& error) {
        std::cerr << "Cortex native inference failed: " << error.what() << '\n';
        set_error(response, 500, "The native model could not complete the request.",
                  "native_inference_failed", "server_error");
      }
    };
  };

  server.Post(base + "/chat/completions", inference_handler(true));
  server.Post(base + "/completions", inference_handler(false));
  server.Post(base + "/embeddings", [&engine, &options](const httplib::Request& request,
                                                         httplib::Response& response) {
    try {
      const auto payload = parse_body(request);
      if (!options.embedding) {
        throw ApiError(422, "unsupported_model_capability", "Embeddings are disabled for this model.");
      }
      if (payload.contains("stream") &&
          (!payload.at("stream").is_boolean() || payload.at("stream").get<bool>())) {
        throw ApiError(400, "invalid_request", "Embedding responses cannot be streamed.");
      }
      if (!payload.contains("input")) {
        throw ApiError(400, "invalid_request", "input is required for embeddings.");
      }
      std::vector<std::string> inputs;
      const auto& input = payload.at("input");
      if (input.is_string()) {
        inputs.push_back(get_text(input, "input"));
      } else if (input.is_array() && !input.empty()) {
        inputs.reserve(input.size());
        for (const auto& item : input) {
          inputs.push_back(get_text(item, "input item"));
        }
      } else {
        throw ApiError(400, "invalid_request", "input must be a string or a non-empty string array.");
      }

      Json rows = Json::array();
      std::size_t total_tokens = 0;
      for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto tokens = engine.tokenize(inputs[index]);
        auto embedding = engine.embed(tokens);
        total_tokens += tokens.size();
        rows.push_back({{"object", "embedding"}, {"index", index}, {"embedding", std::move(embedding)}});
      }
      Json result = {{"object", "list"},
                     {"data", std::move(rows)},
                     {"model", options.model_alias},
                     {"usage", {{"prompt_tokens", total_tokens}, {"total_tokens", total_tokens}}}};
      response.set_content(result.dump(-1, ' ', false, Json::error_handler_t::replace),
                           "application/json");
    } catch (const ApiError& error) {
      set_error(response, error.status(), error.what(), error.code());
    } catch (const std::exception& error) {
      std::cerr << "Cortex native embedding failed: " << error.what() << '\n';
      set_error(response, 500, "The native model could not complete the embedding request.",
                "native_inference_failed", "server_error");
    }
  });
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    Engine engine(options);
    httplib::Server server;
    server.set_payload_max_length(kMaxRequestBytes);
    server.set_read_timeout(10, 0);
    server.set_write_timeout(300, 0);
    server.new_task_queue = [] { return new httplib::ThreadPool(4); };
    register_routes(server, engine, options, engine.model());
    std::cerr << "Cortex native GGUF backend ready on 127.0.0.1:" << options.port << '\n';
    if (!server.listen(options.host.c_str(), static_cast<int>(options.port))) {
      std::cerr << "Cortex native GGUF backend failed to bind its loopback port.\n";
      return 2;
    }
  } catch (const std::exception& error) {
    std::cerr << "Cortex native GGUF backend startup failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
