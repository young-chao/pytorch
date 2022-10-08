此文件夹包含 c10 调度程序。 这个调度器是一个单点，我们计划通过它路由所有内核调用。 
计划替换旧版 PyTorch 或 caffe2 的现有调度机制。

此文件夹包含以下文件：

- Dispatcher.h：主接口。 使用调度程序的代码应该只使用它。
- DispatchTable.h：实际调度机制的实现。 带有内核的哈希表，查找，...
- KernelFunction.h：调用内核的核心接口（即函数指针）

This folder contains the c10 dispatcher. This dispatcher is a single point
through which we are planning to route all kernel calls.
Existing dispatch mechanisms from legacy PyTorch or caffe2 are planned to
be replaced.

This folder contains the following files:
- Dispatcher.h: Main facade interface. Code using the dispatcher should only use this.
- DispatchTable.h: Implementation of the actual dispatch mechanism. Hash table with kernels, lookup, ...
- KernelFunction.h: The core interface (i.e. function pointer) for calling a kernel
