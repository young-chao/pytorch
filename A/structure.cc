torch/csrc
├── /autograd
│   ├── autograd.h/cpp                 # 自动梯度方法，包含backward等方法，由VariableHooks中方法调用
│   ├── custom_function.h/cpp
│   |   ├── AutogradContext            # 负责存储在前向过程中产生的信息，是autograd中操作的上下文
│   ├── edge.h
│   |   ├── Edge                       # 计算图中的边，Node和序号组成的二元组{grad_fn, input_nr}
│   ├── engine.h/cpp
│   |   ├── NodeTask                   # 单个节点计算任务信息定义
│   |   ├── ReadyQueue                 # 存储NodeTask的就绪队列
│   |   ├── Engine                     # 执行自动梯度计算过程的引擎，如autograd过程需调用其execute方法启动
│   ├── function.h/cpp                 # 定义由节点生成边、获取边的方法
│   |   ├── Node                       # 计算图中的节点，其apply方法是非叶节点向叶节点方向操作的反向传播函数
│   ├── graph_task.h
│   |   ├── GraphTask                  # 计算图，含有autograd过程的整体信息
│   ├── input_buffer.h
│   |   ├── InputBuffer                # 存储节点输入
│   ├── python_variable.h/cpp
│   |   ├── THPVariable                # THPVariableType实际对应的类（成员：at::Tensor指针）
│   |   ├── THPVariableType            # 即python代码中的torch._C._TensorBase/torch.autograd.Tensor
│   ├── variable.h/cpp
│   |   ├── Variable                   # at::Tensor的别名          
│   |   ├── AutogradMeta               # autograd过程元信息，继承c10::AutogradMetaInterface
│   |   ├── ViewInfo                   # 视图信息
│   |   ├── DifferentiableViewMeta     # 可微视图，包含前向、后向视图信息
│   |   ├── VariableHooks              # Variable的hook方法实现，继承at::impl::VariableHooksInterface
│   ├── /function
│   │   ├── basic_ops.h
│   │   |   ├── GraphRoot              # autograd过程的根节点定义，若roots含多个节点则构建一个虚拟的根节点
│   ├── ... ...   


aten/src/ATen
├── /core
│   ├── TensorBase.h        
│   |   ├── TensorBase                 # Tensor基本定义，封装调用TensorImpl的方法（成员：c10::TensorImpl指针）
│   ├── Tensor.h/cpp                   # Tensor类和TensorBase类的hook相关方法实现
│   ├── VariableHooksInterface.h
│   |   ├── VariableHooksInterface     # Hook方法接口，Tensor的成员方法中均嵌入该接口方法的调用
├── /templates
│   ├── TensorBody.h
│   |   ├── Tensor                     # Tensor，继承TensorBase，重载运算符、梯度等信息获取、hook相关方法
│   ├── ... ...   


c10
├── /core
│   ├── Allocator.h/cpp
│   |   ├── Allocator                  # 分配器，用于给StorageImpl对象分配真实存储
│   |   ├── DataPtr                    # 内存数据存储，包含指向内存的指针和设备信息
│   ├── Storage.h/cpp
│   |   ├── Storage                    # Tensor存储（成员：StorageImpl指针）
│   ├── StorageImpl.h/cpp
│   |   ├── StorageImpl                # Tensor存储功能实现，继承intrusive_ptr_target（成员：DataPtr、Allocator指针）
│   ├── TensorImpl.h/cpp
│   |   ├── AutogradMetaInterface      # 记录自动梯度过程元信息的接口
│   |   ├── TensorImpl                 # Tensor实现，继承intrusive_ptr_target（成员：Storage、AutogradMetaInterface指针）
│   ├── ... ...          
