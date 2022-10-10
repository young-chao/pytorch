// 由function注册到kernel实现(nn.functional.hardsigmoid)
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
/*
定义内核函数，然后使用DECLARE/ register DISPATCH对其进行注册，以cpu为例过程如下:
1) 在头文件中使用DECLARE_DISPATCH(fn_type, fnNameImpl)声明调度;其中fn_type是内核的函数指针类型(例如，定义为使用
   fn_type = void(*)(tenor &，const tenor &), fnNameImpl是调度注册表的名称。
2) 使用DEFINE_DISPATCH(fnNameImpl)(与声明的名称匹配)在cpu目录之外的c++文件中定义调度(调度必须只定义一次)。在这个
   c++文件中包含声明分派的头文件。按照惯例，在定义native函数的文件中定义调度。
3) 定义一个使用fnNameImpl(kCPU, arguments…)调用调度的native函数，其中参数是根据在声明中定义的fn_type定义的参数。
4) 将实际内核(例如your_kernel)写入cpu目录，并使用REGISTER_DISPATCH(fnNameImpl，&your_kernel)将其注册到调度。
*/
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

// 定义调度实现
DEFINE_DISPATCH(hardsigmoid_stub);
DEFINE_DISPATCH(hardsigmoid_backward_stub);
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
/* 
   aten/src/ATen/native/Activation.h:
*/
using hardsigmoid_fn = void(*)(TensorIteratorBase&);
using hardsigmoid_backward_fn = void(*)(TensorIteratorBase&);

// fn为函数指针，stub为调度注册表
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
