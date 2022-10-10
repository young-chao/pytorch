// 算子由function注册到kernel调用(nn.functional.hardsigmoid)
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
/* 
   aten/src/ATen/native/native_functions.yaml:
   函数注册，dispatch针对不同的backend调用不同的kernel
*/
- func: hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  device_check: NoCheck   # TensorIterator
  python_module: nn
  dispatch:
    CPU, CUDA: hardsigmoid_out
    QuantizedCPU: hardsigmoid_out_quantized_cpu
    
- func: hardsigmoid(Tensor self) -> Tensor
  structured_delegate: hardsigmoid.out
  device_check: NoCheck   # TensorIterator
  python_module: nn
  dispatch:
    QuantizedCPU: hardsigmoid_quantized_cpu

- func: hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  python_module: nn
  dispatch:
    CPU, CUDA: hardsigmoid_backward_out

- func: hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor
  structured_delegate: hardsigmoid_backward.grad_input
  python_module: nn
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
/* 
   tools/autograd/derivatives.yaml:
   backward函数注册
*/
- name: hardsigmoid(Tensor self) -> Tensor
  self: hardsigmoid_backward(grad, self)
  result: auto_element_wise
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
/* 
   aten/src/ATen/native/Activation.cpp:
   激活函数位于Activation.cpp，其它函数位于该目录下其它文件
*/
TORCH_IMPL_FUNC(hardsigmoid_out) (
  const Tensor& self, const Tensor& result
) {
  hardsigmoid_stub(device_type(), *this);
}
TORCH_IMPL_FUNC(hardsigmoid_backward_out) (
  const Tensor& grad_output, const Tensor& self, const Tensor& grad_input
) {
  hardsigmoid_backward_stub(device_type(), *this);
}
/*
TORCH_IMPL_FUNC(hardsigmoid_out) 等价为 void structured_hardsigmoid_out::impl
TORCH_IMPL_FUNC(hardsigmoid_backward_out) 等价为 void structured_hardsigmoid_backward_out::impl
*/

DEFINE_DISPATCH(hardsigmoid_stub);
DEFINE_DISPATCH(hardsigmoid_backward_stub);
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
/* 
   aten/src/ATen/native/Activation.h:
*/
using hardsigmoid_fn = void(*)(TensorIteratorBase&);
using hardsigmoid_backward_fn = void(*)(TensorIteratorBase&);

DECLARE_DISPATCH(hardsigmoid_fn, hardsigmoid_stub);
DECLARE_DISPATCH(hardsigmoid_backward_fn, hardsigmoid_backward_stub);
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
/* 
   aten/src/ATen/native/cpu/Activation.cpp:
   cpu后端的kernel函数实现
*/
REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel);
REGISTER_DISPATCH(hardsigmoid_backward_stub, &hardsigmoid_backward_kernel);

void hardsigmoid_kernel(TensorIteratorBase& iter) {...}
void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {...}
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
