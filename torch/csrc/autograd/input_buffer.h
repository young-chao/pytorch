#pragma once

// The InputBuffer class accumulates a list of Variables for use by a
// function. It implements logic to avoid modifying the passed
// values in-place (adding an input twice will accumulate the result).
// This behaviour is needed and used only in backward graphs.

#include <memory>
#include <utility>
#include <vector>

#include <c10/core/Stream.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/variable.h>

namespace torch {
namespace autograd {

// 一些节点在反向计算时有多个输入。因此在计算梯度的时候，grad_fn的输入可能从
// 很多条路径上累积过来，InputBuffer就是用来累加grad_fn的输入。
struct InputBuffer {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit InputBuffer(size_t size) : buffer(size) {} //size表示输入数量
  InputBuffer(const InputBuffer& other) = delete;
  InputBuffer(InputBuffer&& other) = default;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit InputBuffer(variable_list&& inputs) : buffer(std::move(inputs)){};
  InputBuffer& operator=(InputBuffer&& other) = default;

  // 累积指定索引处的variable，若为空则直接添加
  // Accumulates the variable at a specified index.
  // The optional CUDA streams determine which stream the accumulation
  // is run on and how the addition is synchronized.
  void add(
      size_t pos,
      Variable&& var,
      const c10::optional<c10::Stream>& opt_producer_stream,
      const c10::optional<c10::Stream>& opt_consumer_stream);

  at::Device device() const; //获取输入张量的设备

  // 重载[]用于取对应位置的梯度Variable
  Variable operator[](size_t pos) {
    return buffer[pos];
  }

  // Returns the inputs as a list of variables. Destroys given InputBuffer.
  static std::vector<Variable> variables(InputBuffer&& g);

 private:
  std::vector<Variable> buffer; //存储输入variable
};

} // namespace autograd
} // namespace torch
