#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cortex::native {

template <typename Slot>
class InferenceSlotPool {
 public:
  class Lease {
   public:
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    Lease(Lease&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_) {}

    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        reset();
        owner_ = std::exchange(other.owner_, nullptr);
        index_ = other.index_;
      }
      return *this;
    }

    ~Lease() { reset(); }

    Slot& get() const { return *owner_->slots_[index_]; }
    Slot& operator*() const { return get(); }
    Slot* operator->() const { return &get(); }

   private:
    friend class InferenceSlotPool<Slot>;

    Lease(InferenceSlotPool* owner, std::size_t index) noexcept
        : owner_(owner), index_(index) {}

    void reset() noexcept {
      if (owner_ != nullptr) {
        owner_->release(index_);
        owner_ = nullptr;
      }
    }

    InferenceSlotPool* owner_ = nullptr;
    std::size_t index_ = 0;
  };

  explicit InferenceSlotPool(std::vector<std::unique_ptr<Slot>> slots)
      : slots_(std::move(slots)), occupied_(slots_.size(), 0) {
    if (slots_.empty() ||
        std::any_of(slots_.begin(), slots_.end(), [](const auto& slot) { return !slot; })) {
      throw std::invalid_argument("An inference slot pool needs non-null slots.");
    }
  }

  InferenceSlotPool(const InferenceSlotPool&) = delete;
  InferenceSlotPool& operator=(const InferenceSlotPool&) = delete;

  Lease acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    available_.wait(lock, [this] {
      return std::any_of(occupied_.begin(), occupied_.end(),
                         [](std::uint8_t occupied) { return occupied == 0; });
    });
    for (std::size_t index = 0; index < occupied_.size(); ++index) {
      if (occupied_[index] == 0) {
        occupied_[index] = 1;
        return Lease(this, index);
      }
    }
    throw std::logic_error("Inference slot availability changed unexpectedly.");
  }

  std::size_t size() const noexcept { return slots_.size(); }

 private:
  void release(std::size_t index) noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      occupied_[index] = 0;
    }
    available_.notify_one();
  }

  std::vector<std::unique_ptr<Slot>> slots_;
  std::vector<std::uint8_t> occupied_;
  std::mutex mutex_;
  std::condition_variable available_;
};

}  // namespace cortex::native
