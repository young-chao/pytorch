#pragma once
#include <memory>

#include <c10/macros/Macros.h>

namespace c10 {

using DeleterFnPtr = void (*)(void*);

namespace detail {

// Does not delete anything
TORCH_API void deleteNothing(void*);

// detail::UniqueVoidPtr是一个类似于unique_ptr的所有权智能指针，但有三个主要区别：
// A detail::UniqueVoidPtr is an owning smart pointer like unique_ptr, but
// with three major differences:
//
//    1) 专门用于void
//    1) It is specialized to void
//
//    2) 专门用于函数指针删除器 void(void* ctx); 即，删除器不引用数据，只引用上下文指针（被删除为 void*）。 事实上，在内部，这个指针被实现为拥有对上下文的引用和对数据的非拥有引用； 这就是为什么您使用 release_context() 而不是 release() 的原因（用于 release() 的传统 API 不会为您提供足够的信息来稍后正确处理该对象。）
//    2) It is specialized for a function pointer deleter
//       void(void* ctx); i.e., the deleter doesn't take a
//       reference to the data, just to a context pointer
//       (erased as void*).  In fact, internally, this pointer
//       is implemented as having an owning reference to
//       context, and a non-owning reference to data; this is why
//       you release_context(), not release() (the conventional
//       API for release() wouldn't give you enough information
//       to properly dispose of the object later.)
//
//    3) 保证在唯一指针被销毁且上下文非空时调用deleter. 这与std::unique_ptr不同, 
//       std::unique_ptr如果数据指针为空，则不调用删除器. 
//    3) The deleter is guaranteed to be called when the unique
//       pointer is destructed and the context is non-null; this is different
//       from std::unique_ptr where the deleter is not called if the
//       data pointer is null.
//
// Some of the methods have slightly different types than std::unique_ptr
// to reflect this.
//
class UniqueVoidPtr {
 private:
  // Lifetime tied to ctx_
  void* data_;
  std::unique_ptr<void, DeleterFnPtr> ctx_;

 public:
  UniqueVoidPtr() : data_(nullptr), ctx_(nullptr, &deleteNothing) {}
  explicit UniqueVoidPtr(void* data)
      : data_(data), ctx_(nullptr, &deleteNothing) {}
  UniqueVoidPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter)
      : data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &deleteNothing) {}
  void* operator->() const {
    return data_;
  }
  void clear() {
    ctx_ = nullptr;
    data_ = nullptr;
  }
  void* get() const {
    return data_;
  }
  void* get_context() const {
    return ctx_.get();
  }
  void* release_context() {
    return ctx_.release();
  }
  std::unique_ptr<void, DeleterFnPtr>&& move_context() {
    return std::move(ctx_);
  }
  C10_NODISCARD bool compare_exchange_deleter(
      DeleterFnPtr expected_deleter,
      DeleterFnPtr new_deleter) {
    if (get_deleter() != expected_deleter)
      return false;
    ctx_ = std::unique_ptr<void, DeleterFnPtr>(ctx_.release(), new_deleter);
    return true;
  }

  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    if (get_deleter() != expected_deleter)
      return nullptr;
    return static_cast<T*>(get_context());
  }
  operator bool() const {
    return data_ || ctx_;
  }
  DeleterFnPtr get_deleter() const {
    return ctx_.get_deleter();
  }
};

// Note [How UniqueVoidPtr is implemented]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UniqueVoidPtr 解决了张量数据分配器的一个常见问题，即您感兴趣的数据指针（例如, float*）
// 与您实际解除分配数据所需的上下文指针（例如，DLManagedTensor）不同. 在传统的删除器设计
// 下，您必须在删除器本身中存储额外的上下文，以便您可以实际删除正确的内容。
// 用标准 C++ 实现这个有点容易出错：如果使用 std::unique_ptr 来管理张量，如果数据指针为 
// nullptr，则不会调用删除器，如果上下文指针为非null(并且删除器负责释放数据指针和上下文
// 指针)，则可能导致泄漏. 
// UniqueVoidPtr solves a common problem for allocators of tensor data, which
// is that the data pointer (e.g., float*) which you are interested in, is not
// the same as the context pointer (e.g., DLManagedTensor) which you need
// to actually deallocate the data.  Under a conventional deleter design, you
// have to store extra context in the deleter itself so that you can actually
// delete the right thing.  Implementing this with standard C++ is somewhat
// error-prone: if you use a std::unique_ptr to manage tensors, the deleter will
// not be called if the data pointer is nullptr, which can cause a leak if the
// context pointer is non-null (and the deleter is responsible for freeing both
// the data pointer and the context pointer).
//
// 因此，在我们重新实现的 unique_ptr 中，它就将上下文直接存储在唯一指针中，并将删
// 除器附加到上下文指针本身。 在简单的情况下，上下文指针就是指针本身。
// So, in our reimplementation of unique_ptr, which just store the context
// directly in the unique pointer, and attach the deleter to the context
// pointer itself.  In simple cases, the context pointer is just the pointer
// itself.

inline bool operator==(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return !sp;
}
inline bool operator==(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return !sp;
}
inline bool operator!=(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return sp;
}
inline bool operator!=(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return sp;
}

} // namespace detail
} // namespace c10
