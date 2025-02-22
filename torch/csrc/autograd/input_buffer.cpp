#include <torch/csrc/autograd/input_buffer.h>

#include <ATen/BatchedTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/TensorOperators.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace autograd {

namespace {
// look what you made me do >.<
// Divergent paths for per-Impl stream recording that leak implementation
// details of the impls should not be needed here.
// See https://github.com/pytorch/pytorch/issues/60306
// TODO: clean this up when https://github.com/pytorch/pytorch/issues/60306 is
// improved
void record_stream_any_impl(Variable& var, c10::Stream& stream) {
  const auto guard = c10::impl::VirtualGuardImpl(c10::DeviceType::CUDA);

  if (C10_UNLIKELY(at::isBatchedTensor(var))) {
    auto* impl = at::maybeGetBatchedImpl(var);
    if (impl) {
      guard.recordDataPtrOnStream(impl->value().storage().data_ptr(), stream);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Expected batched tensor");
    }
  } else {
    switch (var.layout()) {
      case c10::kSparseCsr:
      case c10::kSparseCsc:
      case c10::kSparseBsr:
      case c10::kSparseBsc: {
        auto* impl = at::sparse_csr::get_sparse_csr_impl(var);
        guard.recordDataPtrOnStream(
            impl->values().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->compressed_indices().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->plain_indices().storage().data_ptr(), stream);
        break;
      }
      case c10::kSparse: {
        auto* impl = at::sparse::get_sparse_impl(var);
        guard.recordDataPtrOnStream(
            impl->values().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->indices().storage().data_ptr(), stream);
        break;
      }
      case c10::kStrided:
        guard.recordDataPtrOnStream(var.storage().data_ptr(), stream);
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Unknown layout in record_stream_any_impl");
    }
  }
}
} // anonymous namespace

// 就地累加梯度
static void accumulate(
    std::vector<Variable>& buffer,
    const size_t pos,
    Variable&& var) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
  auto& old_var = buffer[pos];
  // ATen doesn't route sparse additions correctly...
  // do dense + sparse in-place if possible
  // 当某一方为COO矩阵表示的稀疏张量时，需先进行转化再相加（add_中已实现转化）
  // 两者都是稀疏张量或非稀疏张量则直接相加
  if (old_var.is_sparse()) {
    // It is safe to change the Tensor inplace if the Tensor is only used in
    // this buffer (this could be the gradient passed by the user) and that no
    // other Tensor is using the same storage.
    if (!var.is_sparse() && var.is_contiguous() && var.use_count() == 1 &&
        var.storage().use_count() == 1) {
      buffer[pos] = var.add_(old_var);
    } else {
      buffer[pos] = var + old_var;
    }
  } else {
    if (var.is_sparse() && !old_var.is_sparse() && old_var.is_contiguous() &&
        old_var.use_count() == 1 && old_var.storage().use_count() == 1) {
      buffer[pos] = old_var.add_(var);
    } else {
      buffer[pos] = old_var + var;
    }
  }
}

// InputBuffer累积指定索引处的variable，若为空则直接添加
void InputBuffer::add(
    size_t pos,
    Variable&& var,
    const c10::optional<c10::Stream>& opt_producer_stream,
    const c10::optional<c10::Stream>& opt_consumer_stream) {
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
  if (!var.defined()) {
    return;
  }

  /* 切换到累积设备
     选择用于累积的设备（和流）是：
     (1) var不是CUDA变量。累积发生在var的设备上。
     (2) var是一个CUDA变量，它、消费者和生产者共享同一个设备：
         (2a) 使用消费者的流作为累积流 
         (2b) 将累积流与生产者的流同步（如果不同）
         (2c) 累积
     (3) var是一个CUDA变量，它与消费者共享一个设备，但不与生产者共享： 
         (3a) 使用消费者的流作为累积流 
         (3b) 将累积流与消费者设备的默认流同步 
         (3c) 累积。
     (4) var是一个CUDA变量，它与生产者共享一个设备，但不与消费者共享一个设备： 
         (4a) 使用生产者设备的默认流作为累积流 
         (4b) 将累积流与生产者的流同步 
         (4c) 累积.
     (5) var是一个CUDA变量，它不与消费者或生产者共享设备。累积发生在 var 设备的默认流上。
  */
  // Switches to accumulate device
  // The device (and stream) chosen for accumulation is:
  //  (1) var is not a CUDA variable. Accumulation happens on var's device.
  //  (2) var is a CUDA variable and it, the consumer, and the producer share
  //  the same device:
  //       (2a) Uses the consumer's stream as the accumulation stream
  //       (2b) Syncs the accumulation stream with the producer's stream (if
  //       different) (2c) Accumulates.
  //  (3) var is a CUDA variable and it shares a device with the consumer but
  //  not the producer:
  //       (3a) Uses the consumer's stream as the accumulation stream
  //       (3b) Syncs the accumulation stream with the consumer device's default
  //       stream (3c) Accumulates.
  //  (4) var is a CUDA variable and it shares a device with the producer but
  //  not the consumer:
  //       (4a) Uses the producer device's default stream as the accumulation
  //       stream (4b) Syncs the accumulation stream with the the producer's
  //       stream (4c) Accumulates.
  //  (5) var is a CUDA variable and it does not share a device with the
  //  consumer or producer.
  //      Accumulation happens on the var device's default stream.

  TORCH_INTERNAL_ASSERT(device_of(var));
  c10::optional<c10::Stream> opt_accumulate_stream = c10::nullopt;
  if (device_of(var)->is_cuda()) {
    const auto on_producer =
        opt_producer_stream && device_of(var) == opt_producer_stream->device();
    const auto on_consumer =
        opt_consumer_stream && device_of(var) == opt_consumer_stream->device();

    if (on_producer && on_consumer) {
      // (2a)
      opt_accumulate_stream = opt_consumer_stream;
      if (opt_accumulate_stream != opt_producer_stream) {
        // (2b)
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(*opt_producer_stream);
        opt_accumulate_stream->wait(event);
        record_stream_any_impl(var, *opt_accumulate_stream);
      }
    } else {
      c10::optional<c10::Stream> opt_sync_stream = c10::nullopt;
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      if (on_consumer && !on_producer) {
        // (3a)
        opt_accumulate_stream = opt_consumer_stream;
        opt_sync_stream = guard.getDefaultStream(opt_consumer_stream->device());
      } else if (on_producer && !on_consumer) {
        // (4a)
        opt_accumulate_stream =
            guard.getDefaultStream(opt_producer_stream->device());
        opt_sync_stream = opt_producer_stream;
      } else {
        // (5)
        opt_accumulate_stream = guard.getDefaultStream(*device_of(var));
      }
      if (opt_sync_stream && (opt_accumulate_stream != opt_sync_stream)) {
        // (3b), (4b)
        c10::OptionalDeviceGuard device_guard{opt_sync_stream->device()};
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(*opt_sync_stream);
        opt_accumulate_stream->wait(event);
        const auto guard = c10::impl::VirtualGuardImpl(c10::DeviceType::CUDA);
        record_stream_any_impl(var, *opt_accumulate_stream);
      }
    }
  }

  auto& old_var = buffer[pos];
  if (!old_var.defined()) {
    buffer[pos] = std::move(var);
  } else {
    if (opt_accumulate_stream) {
      c10::OptionalStreamGuard stream_guard{opt_accumulate_stream};
      accumulate(buffer, pos, std::move(var));
    } else {
      // (1) non-CUDA variable
      //     Accumulation happens on variable's device
      c10::OptionalDeviceGuard device_guard{device_of(var)};
      accumulate(buffer, pos, std::move(var));
    }
  }
}

// 遍历input_buffer中的variables，其中第一个设备非cpu的variable的device将成为
// input_buffer的device，否则设备就是CPU。
auto InputBuffer::device() const -> at::Device {
  // Since we pick the first non-CPU tensor, this won't work with
  // mixed device-type operations (e.g., an op that is both CUDA
  // and XLA).  This is *incredibly* unlikely, so we don't worry
  // about it.
  for (auto& var : buffer) {
    if (var.defined()) {
      auto device = var.device();
      if (device.type() != at::kCPU) {
        return device;
      }
    }
  }
  // Only report to the CPU thread if there really were no tensors
  // from other devices.
  return at::kCPU;
}

auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);
  return result;
}

} // namespace autograd
} // namespace torch
