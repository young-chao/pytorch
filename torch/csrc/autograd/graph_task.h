#pragma once
#include <ATen/ThreadLocalState.h>
#include <ATen/core/Tensor.h>
#include <c10/util/ThreadLocal.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/autograd/utils/warnings.h>
#include <vector>

namespace torch {
namespace autograd {

using edge_list = std::vector<Edge>;
struct ReadyQueue;

static constexpr int NO_DEVICE = -2;
static constexpr int CPU_DEVICE = -1;

// GraphTask实例代表一个动态图级别的资源管理对象，拥有一次反向传播执行所需要的全部元数据。
// 如果允许重入反向传播，则会有多个GraphTask一起工作。
// GraphTask holds metadata needed for a single execution of backward()
struct GraphTask : std::enable_shared_from_this<GraphTask> {
  std::atomic<uint64_t> outstanding_tasks_{0}; //用来记录当前任务数目，如果数目为0，则说明任务结束。 如果这个数量不为0，则此GraphTask依然需要运行。
  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error_{false}; //错误发生标识符
  std::atomic_bool future_completed_{false};
  // It is safe to read keep_graph_ without synchronization
  bool keep_graph_; //用来指定一次反向计算后是否释放资源。

  // To protect reads/writes to not_ready_, dependencies_, captured_vars_,
  // has_error_, future_result_, cpu_ready_queue_, and leaf_streams.
  std::mutex mutex_; //互斥量，用于保护对以上英文注释所提及变量的读写
  std::unordered_map<Node*, InputBuffer> not_ready_; //针对未就绪节点和其输入的map。key是未就绪节点，value是这个节点目前就绪的输入列表。
  std::unordered_map<Node*, int> dependencies_; //记录节点间的依赖，用来判断后续节点是否已经可以被执行。

  /* Exec info允许过滤图上不需要的路径。
     如果它为空，则意味着任务以默认模式运行，所有next_edges都应该被执行。 
     如果它不为空，则仅执行exec_info[node].needed_=true的特定节点。
     这些节点的性质是：节点拥有一条路径，这路径可以通往outputs的任何一条边。
     exec_info仅在通过.backward()执行图形且未传递输入参数时为空。
     当通过.grad()执行或为.backward()指定inputs时，exec_info将非空。
     */
  // Note [Exec info]
  // Exec info is created for each GraphTask, which allows filtering paths on
  // the graph that are not needed. It has a bit complicated semantics. If it's
  // empty, it means the task is run in a "default" mode, which means that all
  // next_edges we encounter should get executed. If it's not empty, only
  // functions that have an entry and this entry has needed == True should be
  // executed. exec_info is only empty when the graph is executed via
  // .backward() and the inputs parameter is not passed. Otherwise, when
  // executed through .grad(), or when inputs arg is specified for .backward(),
  // exec_info will be non-empty.
  //
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct ExecInfo {
    struct Capture {
      Capture(const Capture&) = delete;
      Capture(Capture&&) = default;

      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      Capture(int input_idx, int output_idx)
          : input_idx_(input_idx), output_idx_(output_idx) {}
      int input_idx_; // within Node inputs
      int output_idx_; // within the output vector of a GraphTask

      // This hook will be executed after a grad is captured. The captured
      // grad will be replaced by the return value of the hook.
      struct GradCaptureHook {
        virtual ~GradCaptureHook() = default;
        virtual at::Tensor operator()(const at::Tensor& grad) = 0;
      };
      // hooks将按照添加的顺序被一一调用, hook的输入grad将是其前一个hook的输出。 
      // 第一个hook会将捕获的grad作为输入, 最后一个hook的输出将替换捕获的grad。
      // The hooks will be called one by one in the order as they were added.
      // The input grad of a hook will be the output of its preceding hook. The
      // first hook will take the captured grad as the input. The output of the
      // last hook will replace the captured grad.
      std::vector<std::unique_ptr<GradCaptureHook>> hooks_;
    };

    bool should_execute() const {
      return needed_ || captures_; //needed和captures均可表示该节点应该执行
    }

    bool needed_ = false; //判断节点是否需要执行的标志符
    std::unique_ptr<std::vector<Capture>> captures_; //判断是否应该返回该节点的梯度
  };
  // exec_info_ is safe to read without synchronization
  std::unordered_map<Node*, ExecInfo> exec_info_; //存储节点执行信息的键值对，用于剪除不需要执行的分支
  // Captures variables are grads captured that we return to the user. After
  // execution of the GraphTask is completed, the captured_vars_ are moved
  // out of the GraphTask and are no longer valid.
  std::vector<Variable> captured_vars_; //用于返回给用户的梯度Variable

  // Note: this field is not ready to be used until the proper
  // `thread_locals_.set_grad_mode()` call in the constructor.
  at::ThreadLocalState thread_locals_ = at::ThreadLocalState();

  std::unordered_set<c10::Stream> leaf_streams;

  // Per-device current streams of the execute() that called this GraphTask.
  // These will be synced with leaf_streams in exec_post_processing.
  std::vector<c10::optional<c10::Stream>> caller_current_streams_;

  // Collects caller_current_streams_
  void stash_current_streams();

  void init_to_execute(
      Node& graph_root,
      const edge_list& outputs,
      bool accumulate_grad,
      uint64_t min_topo_nr);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  // Safe to read owner_ and reentrant_depth_ without synchronizaton
  int owner_; //GraphTask所属线程的Device数值。该值是创建GraphTask线程中的worker_device的值。
  // The number of parent graph tasks for this graph task
  const int reentrant_depth_;

  bool can_checkpoint() const {
    return exec_info_.empty();
  }

  // check if the GraphTask is completed or not
  bool completed();
  // mark the graph task as completed and trigger post processing
  void mark_as_completed_and_run_post_processing();

  // Set an appropriate exception on this graph_task which was encountered while
  // running the provided function.
  void set_exception(std::exception_ptr eptr, const std::shared_ptr<Node>& fn);

  // Set an appropriate exception on this graph_task which was encountered while
  // running the provided function. But doesn't signal completion on
  // 'future_result_' right away. The user needs to explicitly mark
  // 'future_result_' completed with an appropriate exception.
  void set_exception_without_signal(const std::shared_ptr<Node>& fn);

  // Whether or not to stop execution for this GraphTask when an error is
  // encountered. When set to true, this would cause Engine::execute() to throw
  // an exception as soon as the autograd engine receives an exception.
  bool exit_on_error_;

  /* CPU线程专用于处理反向传播之中的CPU相关工作。因此所有GraphTask都会维护自己的
     cpu_ready_queue_，CPU相关任务应该发送到该队列。对于每个GraphTask，我们维护
     cpu_ready_queue_，这样在设备线程（即GPU）上执行时，如果下一个NodeTask应该在
     CPU上运行，我们就知道应该推送NodeTask到哪个就绪队列。*/
  // CPU threads are dedicated to processing CPU work for the backward they
  // invoked. So any given graph task maintains its own cpu_ready_queue_ where
  // you should send work for it to be done. We memoize the cpu_ready_queue_ per
  // GraphTask so that we know which ready queue we should push to if we are on
  // device thread (i.e. GPU) and but next NodeTask should be run on CPU.
  std::shared_ptr<ReadyQueue> cpu_ready_queue_; //CPU线程队列

  // Future representing the completion of the graph task. Notified when all
  // tasks are done.
  c10::intrusive_ptr<at::ivalue::Future> future_result_;

  // Final callbacks installed during execution of this GraphTask
  std::vector<std::function<void()>> final_callbacks_;
  // To protect reads and writes to final_callbacks_. Intentionally no reusing
  // mutex_ as the two are protecting different data structures.
  std::mutex final_callbacks_lock_;

  utils::DelayWarningHandler warning_handler_;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  GraphTask(
      bool keep_graph,
      bool grad_mode,
      int reentrant_depth,
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      bool exit_on_error = false)
      : keep_graph_(keep_graph),
        owner_(NO_DEVICE),
        reentrant_depth_(reentrant_depth),
        exit_on_error_(exit_on_error),
        cpu_ready_queue_(std::move(cpu_ready_queue)),
        future_result_(c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()))) {
    thread_locals_.set_grad_mode(grad_mode);
  }

 private:
  // run GraphTask post processing
  void exec_post_processing();
};

// The guard that sets and restores current_graph_task.
class GraphTaskGuard {
 public:
  explicit GraphTaskGuard(std::shared_ptr<GraphTask> graph_task);
  ~GraphTaskGuard();

  void restore_current_graph_task();

 private:
  std::shared_ptr<GraphTask> last_graph_task_;
};

TORCH_API const std::unordered_map<Node*, GraphTask::ExecInfo>*
get_current_graph_task_exec_info();
TORCH_API bool get_current_graph_task_keep_graph();
void add_node_to_current_graph_task_exec_info(Node* fn);

} // namespace autograd
} // namespace torch
