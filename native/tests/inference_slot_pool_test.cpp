#include "inference_slot_pool.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct TestSlot {
  explicit TestSlot(int value) : value(value) {}
  int value;
};

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  try {
    std::vector<std::unique_ptr<TestSlot>> slots;
    slots.push_back(std::make_unique<TestSlot>(1));
    slots.push_back(std::make_unique<TestSlot>(2));
    cortex::native::InferenceSlotPool<TestSlot> pool(std::move(slots));

    int released_value = 0;
    {
      auto first = pool.acquire();
      auto second = pool.acquire();
      require(pool.size() == 2, "slot-pool size was not preserved");
      require(first->value != second->value, "two leases acquired the same slot");
      released_value = first->value;
    }

    auto reused = pool.acquire();
    require(reused->value == released_value, "a released slot was not made available again");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "Inference slot-pool tests passed.\n";
  return 0;
}
