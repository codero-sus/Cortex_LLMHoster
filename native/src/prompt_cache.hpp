#pragma once

#include <cstddef>
#include <vector>

namespace cortex::native {

struct PromptCachePlan {
  std::size_t keep_tokens;
  bool clear_memory;
};

// Plan safe reuse of the single cached prompt sequence. When trimming a prior
// sequence, keep one fewer token so the caller can replay the final shared token
// and refresh logits before sampling.
template <typename Token>
PromptCachePlan plan_prompt_cache(const std::vector<Token>& cached,
                                  const std::vector<Token>& incoming) {
  std::size_t shared = 0;
  const auto limit = cached.size() < incoming.size() ? cached.size() : incoming.size();
  while (shared < limit && cached[shared] == incoming[shared]) {
    ++shared;
  }

  if (shared == 0) {
    return {0, true};
  }
  if (shared < cached.size()) {
    const auto keep = shared - 1;
    return {keep, keep == 0};
  }
  return {shared, false};
}

}  // namespace cortex::native
