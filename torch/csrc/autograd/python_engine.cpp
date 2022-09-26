#include <torch/csrc/autograd/python_engine.h>

#include <ATen/BatchedTensorImpl.h>
#include <ATen/VmapMode.h>
#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_saved_variable_hooks.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>

#ifndef _WIN32
#include <pthread.h>
#endif

#include <memory> // for unique_ptr
#include <unordered_set>

using namespace torch::autograd;

struct THPEngine {
  PyObject_HEAD
};

static bool _reinitialize_engine = false;

namespace torch {
namespace autograd {
namespace python {

PythonEngine::PythonEngine() = default;

Engine& PythonEngine::get_python_engine() {
  static PythonEngine engine;
  // This is "probably" thread-safe because the flag is set in a fork handler
  // before any threads are created, and this function is only called with the
  // GIL held. However, using fork + threads is playing with fire so this is
  // more of a "best effort" thing. For example, if the fork occurs while the
  // backwards threads hold a lock, we'll probably deadlock in the engine
  // destructor.
  if (_reinitialize_engine) {
    engine.release_workers();
    engine.~PythonEngine();
    new (&engine) torch::autograd::python::PythonEngine();
    _reinitialize_engine = false;
  }
  return engine;
}

PythonEngine::~PythonEngine() {
  Engine::stop();
}

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 9
#define IS_PYTHON_3_9_PLUS
#endif

void PythonEngine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {
  // Increment thread usage count before acquiring the GIL
  if (should_increment) {
    increment_non_reentrant_thread_count();
  }
  // Create a PyThreadState, but release the GIL. This lets
  // pybind11::gil_scoped_acquire calls inside thread_main acquire the GIL
  // without having to create a new PyThreadState each time.
#if defined(IS_PYTHON_3_9_PLUS) || defined(USE_DEPLOY)
  auto gil = std::make_unique<pybind11::gil_scoped_acquire>();
#else
  pybind11::gil_scoped_acquire gil;
#endif
  pybind11::gil_scoped_release no_gil;
  Engine::thread_init(device, ready_queue, false);

  if (should_increment) {
    // Decrement the count during shutdown if we incremented earlier.
    decrement_non_reentrant_thread_count();
  }

#if defined(IS_PYTHON_3_9_PLUS) || defined(USE_DEPLOY)
  // Do not call PyEval_RestoreThread, PyThreadState_[Clear|DeleteCurrent] if
  // runtime is finalizing
  if (!Py_IsInitialized()) {
    no_gil.disarm();
    // TODO: call disarm rather than leak gil_scoped_acquired once
    // PyThreadState_Clear can safely be called from finalize NOTE: deploy.cpp
    // calls `PyInterpreterState_Delete` to destruct PyThreadState, so avoid
    // use-after-free here.
    gil.release();
  }
#endif
}

void PythonEngine::thread_on_exception(
    std::shared_ptr<GraphTask> graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {
  auto python_err = dynamic_cast<python_error*>(&e);
  if (python_err) {
    python_err->persist();
  }
  Engine::thread_on_exception(graph_task, fn, e);
}

std::unique_ptr<AnomalyMetadata> PythonEngine::make_anomaly_metadata() {
  return std::unique_ptr<AnomalyMetadata>(new PyAnomalyMetadata());
}

std::unique_ptr<SavedVariableHooks> PythonEngine::
    get_default_saved_variable_hooks() {
  return PyDefaultSavedVariableHooks::get_hooks();
}

variable_list PythonEngine::execute(
    const edge_list& roots,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) {
  TORCH_CHECK(
      !PyGILState_Check(),
      "The autograd engine was called while holding the GIL. If you are using the C++ "
      "API, the autograd engine is an expensive operation that does not require the "
      "GIL to be held so you should release it with 'pybind11::gil_scoped_release no_gil;'"
      ". If you are not using the C++ API, please report a bug to the pytorch team.")
  try {
    return Engine::execute(
        roots, inputs, keep_graph, create_graph, accumulate_grad, outputs);
  } catch (python_error& e) {
    e.restore();
    throw;
  }
}

c10::intrusive_ptr<at::ivalue::Future> PythonEngine::execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {
  try {
    return Engine::execute_with_graph_task(
        graph_task, graph_root, std::move(input_buffer));
  } catch (python_error& e) {
    pybind11::gil_scoped_acquire gil;
    if (!PyErr_Occurred()) {
      // Set the error indicator only if it is not set already.
      e.restore();
    }
    throw;
  }
}
} // namespace python
} // namespace autograd
} // namespace torch

PyObject* THPEngineClass = nullptr;

// 计算给定张量相对于图中叶子节点的梯度总和。
// Implementation of torch._C._EngineBase.run_backward
PyObject* THPEngine_run_backward(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 对输入的参数进行解析，初始化相关的变量
  // 若tensor不是一个标量，那么就先构造一个标量的值：L = torch.sum(tensor*grad_tensor)，再关于L对各个叶子张量计算梯度
  PyObject* tensors = nullptr; //准备计算相对叶节点导数的张量. python中使用loss.backward调用时即为loss
  PyObject* grad_tensors = nullptr; //构造出的变量L关于原变量tensor的梯度（默认全1），会以乘积形式影响叶张量的总梯度
  unsigned char keep_graph = 0; //如果为True，用于计算grad的图将不会立刻释放，可以进行多次backward
  unsigned char create_graph = 0; //如果为True，将构造导数图，允许计算高阶导数
  PyObject* inputs = nullptr; //需要计算梯度的张量列表（可包括中间节点），如果为空则计算全部叶子张量
  unsigned char allow_unreachable = 0; //
  unsigned char accumulate_grad = 0; //指示是否将grad累加到叶张量中或捕获
  const char* accepted_kwargs[] = {// NOLINT
                                   "tensors",
                                   "grad_tensors",
                                   "keep_graph",
                                   "create_graph",
                                   "inputs",
                                   "allow_unreachable",
                                   "accumulate_grad",
                                   nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OObb|Obb",
          (char**)accepted_kwargs,
          &tensors,
          &grad_tensors,
          &keep_graph,
          &create_graph,
          &inputs,
          &allow_unreachable,
          &accumulate_grad))
    return nullptr;
  THPUtils_assert(
      PyTuple_Check(tensors),
      "tensors argument is expected to "
      "be a tuple, but got %s",
      THPUtils_typename(tensors));
  THPUtils_assert(
      PyTuple_Check(grad_tensors),
      "grad_tensors argument is "
      "expected to be a tuple, but got %s",
      THPUtils_typename(grad_tensors));

  Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors);
  Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_tensors);
  THPUtils_assert(
      num_tensors == num_gradients,
      "got %ld tensors and %ld "
      "gradients",
      num_tensors,
      num_gradients);
  
  // 调用autograd.backward(...)时为True;调用autograd.grad(...)时为False.
  // The user either called autograd.backward(...) or autograd.grad(...) to get
  // here
  bool backward_api_called = accumulate_grad;
  TORCH_CHECK(
      !backward_api_called || at::impl::VmapMode::current_vmap_level() == 0,
      "backward() called inside torch.vmap. This is not supported, "
      "please call backward() outside torch.vmap or instead use "
      "torch.autograd.grad inside torch.vmap");

  // 把网络最终输出节点的信息放入roots和grads作为后续反向求导的开始
  edge_list roots;
  roots.reserve(num_tensors);
  variable_list grads;
  grads.reserve(num_tensors);
  for (const auto i : c10::irange(num_tensors)) {
    PyObject* _tensor = PyTuple_GET_ITEM(tensors, i);
    THPUtils_assert(
        THPVariable_Check(_tensor),
        "element %d of tensors "
        "tuple is not a Tensor",
        i);
    const auto& variable = THPVariable_Unpack(_tensor); //将THPVariable指针解包为该指针指向实例包含的cdata的引用,即at::Tensor
    TORCH_CHECK(
        !isBatchedTensor(variable),
        "torch.autograd.grad(outputs, inputs, grad_outputs) called inside ",
        "torch.vmap. We do not support the case where any outputs are ",
        "vmapped tensors (output ",
        i,
        " is being vmapped over). Please "
        "call autograd.grad() outside torch.vmap or file a bug report "
        "with your use case.")
    // 获取结果张量的梯度边（梯度边一定唯一，一个张量只能是一个操作的直接输出）
    auto gradient_edge = torch::autograd::impl::gradient_edge(variable);
    THPUtils_assert(
        gradient_edge.function,
        "element %d of tensors does not require grad and does not have a grad_fn",
        i);
    // 梯度边存入roots
    roots.push_back(std::move(gradient_edge));

    PyObject* grad = PyTuple_GET_ITEM(grad_tensors, i);
    if (THPVariable_Check(grad)) {
      const Variable& grad_var = THPVariable_Unpack(grad);
      if (grad_var.has_names()) {
        TORCH_WARN(
            "Autograd was passed a named grad tensor with dims ",
            grad_var.names(),
            ". Autograd does not yet support named tensor semantics, so all names ",
            "will be ignored. In practice all computed gradients will still be correct "
            "according to regular tensor semantics.");
      }
      // 当前tensor相对L的梯度存入grads
      grads.push_back(grad_var);
    } else {
      THPUtils_assert(
          grad == Py_None,
          "element %d of gradients tuple is not a Tensor or None",
          i);
      THPUtils_assert(
          !variable.requires_grad(),
          "element %d of gradients tuple is None, but the corresponding Tensor requires grad");
    }
  }

  std::vector<Edge> output_edges;
  if (inputs != nullptr) { //当inputs给定，记录需要计算梯度的张量的梯度边（含grad_fn或grad_accumulator）
    int num_inputs = PyTuple_GET_SIZE(inputs);
    output_edges.reserve(num_inputs);
    for (const auto i : c10::irange(num_inputs)) {
      PyObject* input = PyTuple_GET_ITEM(inputs, i);
      THPUtils_assert(
          THPVariable_Check(input),
          "all inputs have to be Tensors, but got %s",
          THPUtils_typename(input));
      const auto& tensor = THPVariable_Unpack(input);
      TORCH_CHECK(
          !isBatchedTensor(tensor),
          "torch.autograd.grad(outputs, inputs, grad_outputs) called inside ",
          "torch.vmap. We do not support the case where any inputs are ",
          "vmapped tensors (input ",
          i,
          " is being vmapped over). Please "
          "call autograd.grad() outside torch.vmap or file a bug report "
          "with your use case.")
      const auto output_nr = tensor.output_nr();
      auto grad_fn = tensor.grad_fn();
      if (!grad_fn) {
        grad_fn = torch::autograd::impl::try_get_grad_accumulator(tensor); //如果autograd_meta_为空，则返回null_ptr
      }
      if (accumulate_grad) {
        tensor.retain_grad(); //对非叶节点张量(即中间节点张量)启用用于保存梯度(.grad)
      }
      THPUtils_assert(
          tensor.requires_grad(),
          "One of the differentiated Tensors does not require grad");
      if (!grad_fn) {
        // NOTE [ Autograd Unreachable Input ]
        // Since input has no grad_accumulator, its guaranteed to be
        // unreachable. We initialize an edge pointing to a non-nullptr Node so
        // nodes in the graph (e.g., mul when an operand is scalar) that have
        // edges pointing to nullptr don't get erroneously assigned `needed =
        // True` in exec_info.
        output_edges.emplace_back(std::make_shared<Identity>(), 0); //grad_fn为空，说明为不需要计算梯度的叶张量
      } else {
        output_edges.emplace_back(grad_fn, output_nr);
      }
    }
  }

  variable_list outputs;
  {
    pybind11::gil_scoped_release no_gil;
    auto& engine = python::PythonEngine::get_python_engine();
    outputs = engine.execute(
        roots, grads, keep_graph, create_graph, accumulate_grad, output_edges); //调用python_engine执行backward
  }

  // 调用autograd.grad且指定inputs时返回py_outputs
  // py_outputs为带grad_fn的张量梯度信息
  if (!backward_api_called && inputs != nullptr) {
    int num_inputs = PyTuple_GET_SIZE(inputs);
    THPObjectPtr py_outputs{PyTuple_New(num_inputs)};
    if (!py_outputs)
      return nullptr;
    for (const auto i : c10::irange(num_inputs)) {
      THPUtils_assert(
          allow_unreachable || outputs[i].defined(),
          "One of the "
          "differentiated Tensors appears to not have been used "
          "in the graph. Set allow_unused=True if this is the "
          "desired behavior.");
      PyTuple_SET_ITEM(py_outputs.get(), i, THPVariable_Wrap(outputs[i]));
    }
    return py_outputs.release();
  } else {
    Py_RETURN_NONE; //调用autograd.backward时返回值为None
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_queue_callback(PyObject* self, PyObject* _callback) {
  HANDLE_TH_ERRORS
  auto& engine = python::PythonEngine::get_python_engine();
  std::shared_ptr<PyObject> callback(_callback, [](PyObject* obj) {
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(obj);
  });
  Py_INCREF(_callback);
  engine.queue_callback([callback]() {
    pybind11::gil_scoped_acquire gil;
    THPObjectPtr result{PyObject_CallFunctionObjArgs(callback.get(), nullptr)};
    if (!result)
      throw python_error();
  });
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_is_checkpoint_valid(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto& engine = python::PythonEngine::get_python_engine();
  if (engine.is_checkpoint_valid()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  return type->tp_alloc(type, 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyMethodDef THPEngine_methods[] = {
    {(char*)"run_backward",
     castPyCFunctionWithKeywords(THPEngine_run_backward),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {(char*)"queue_callback", THPEngine_queue_callback, METH_O, nullptr},
    {(char*)"is_checkpoint_valid",
     THPEngine_is_checkpoint_valid,
     METH_NOARGS,
     nullptr},
    {nullptr}};

PyTypeObject THPEngineType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._EngineBase", /* tp_name */
    sizeof(THPEngine), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPEngine_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPEngine_new /* tp_new */
};

static void child_atfork() {
  _reinitialize_engine = true;
}

bool THPEngine_initModule(PyObject* module) {
#ifndef _WIN32
  if (pthread_atfork(nullptr, nullptr, child_atfork) != 0) {
    throw std::runtime_error("unable to set pthread_atfork handler");
  }
#endif
  if (PyType_Ready(&THPEngineType) < 0)
    return false;
  Py_INCREF(&THPEngineType);
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject*)&THPEngineType);
  set_default_engine_stub(python::PythonEngine::get_python_engine);
  return true;
}
