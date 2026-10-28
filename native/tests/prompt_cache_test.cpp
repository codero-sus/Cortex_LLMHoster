#include "prompt_cache.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_plan(const std::string& name, const std::vector<int>& cached,
                 const std::vector<int>& incoming, std::size_t keep_tokens,
                 bool clear_memory) {
  const auto plan = cortex::native::plan_prompt_cache(cached, incoming);
  if (plan.keep_tokens != keep_tokens || plan.clear_memory != clear_memory) {
    throw std::runtime_error(name + ": unexpected prefix-cache plan");
  }
}

}  // namespace

int main() {
  try {
    expect_plan("empty cache", {}, {1, 2}, 0, true);
    expect_plan("exact prompt reuse", {1, 2}, {1, 2}, 2, false);
    expect_plan("prompt extends cache", {1, 2}, {1, 2, 3}, 2, false);
    expect_plan("divergent suffix", {1, 2, 3, 4}, {1, 2, 9}, 1, false);
    expect_plan("shorter prompt", {1, 2, 3, 4}, {1, 2}, 1, false);
    expect_plan("no shared prefix", {1, 2}, {9, 2}, 0, true);
    expect_plan("single-token shared prefix", {1, 2}, {1, 9}, 0, true);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "Prompt-cache planning tests passed.\n";
  return 0;
}
