/*
------------------------------------------------------------------------------------------------
     ********* 结合反向传播的计算图分析 autograd 过程中 Tensor、 Node 和 Edge 的关系 *********
------------------------------------------------------------------------------------------------

————————————————————————————————————————————————————————————————————————————————————————————————
计算过程
a1 = torch.tensor([1., 1.], dtype=torch.float32, requires_grad=True)
a2 = torch.tensor([1., 1.], dtype=torch.float32, requires_grad=True)
a3 = torch.tensor([1., 1.], dtype=torch.float32, requires_grad=True)
b1 = a1+a2
b2 = a2/a3
c = b1*b2
d = torch.mean(c)
————————————————————————————————————————————————————————————————————————————————————————————————

————————————————————————————————————————————————————————————————————————————————————————————————
前向计算图
a1------->
         + = b1------->
a2------->            |
                       * = c-------> mean = d
a2------->            |
         / = b2------->
a3------->

后向计算图
AccumulateGrad(a1) <-------|
                           AddBackward0(b1) <-------|
AccumulateGrad(a2) <-------|                        MulBackward0(c) <-------MeanBackward0(d)
                           DivBackward0(b2) <-------|
AccumulateGrad(a3) <-------|

在前向计算图中，似乎节点是Tensor，而边是Tensor到Tensor的操作。在一些文章中也是这样描述前向计算过程。
但是，结合后向计算图，我们可以发现：计算图中的节点是计算操作，而边是计算操作之间的关联。
--即节点为 MeanBackward0(d)、MulBackward0(c)、AddBackward0(b1)、DivBackward0(b2)、
  AccumulateGrad(a1)、AccumulateGrad(a2)、AccumulateGrad(a3)
--MeanBackward0(d) 到 MulBackward0(c) 的边可以表示为 {MulBackward0(c), 0}
--MulBackward0(c) 到 AddBackward0(b1) 的边可以表示为 {AddBackward0(b1), 0}
  MulBackward0(c) 到 DivBackward0(b2) 的边可以表示为 {DivBackward0(b2), 0}
--AddBackward0(b1) 到 AccumulateGrad(a1) 的边可以表示为 {AccumulateGrad(a1), 0}
  AddBackward0(b1) 到 AccumulateGrad(a2) 的边可以表示为 {AccumulateGrad(a2), 0}
  AddBackward0(b2) 到 AccumulateGrad(a2) 的边可以表示为 {AccumulateGrad(a2), 0}
  AddBackward0(b2) 到 AccumulateGrad(a3) 的边可以表示为 {AccumulateGrad(a3), 0}
————————————————————————————————————————————————————————————————————————————————————————————————

————————————————————————————————————————————————————————————————————————————————————————————————
后向传播autograd过程各Tensor的内容。data为张量值，grad为张量相对于根节点的梯度，grad_fn为张量进行反向传
播用于计算梯度的Node(function)。
--Node实际上是前向传播过程中的操作在backward过程中用于计算梯度的节点
--backward过程中每个Node执行apply方法计算梯度，即 output_grad=apply(input_grad)
--input_grad为当前节点所在张量的梯度
--output_grad为前一个节点所在张量的梯度（的一部分）

a1:
  data: tensor([1., 1.])
  grad: [0.5, 0.5]  //a1.grad = b1.grad_fn(b1.grad)
  grad_fn: None

a2:
  data: tensor([1., 1.])
  grad: [1.5, 1.5]  //a2.grad = b1.grad_fn(b1.grad) + b2.grad_fn(b2.grad)
  grad_fn: None

a3:
  data: tensor([1., 1.])
  grad: [-1., -1.]  //a3.grad = b2.grad_fn(b2.grad)
  grad_fn: None

b1:
  data: tensor([2., 2.])
  grad: [0.5, 0.5]  //b1.grad = c.grad_fn(c.grad)
  grad_fn: AddBackward0

b2:
  data: tensor([1., 1.])
  grad: [1., 1.]  //b2.grad = c.grad_fn(c.grad)
  grad_fn: DivBackward0
  
c:
  data: tensor([2., 2.])
  grad: [0.5, 0.5]  //c.grad = d.grad_fn(d.grad)
  grad_fn: MulBackward0

d:
  data: tensor(2.)
  grad: 1.
  grad_fn: MeanBackward0


————————————————————————————————————————————————————————————————————————————————————————————————

*/
