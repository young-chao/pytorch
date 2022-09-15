#!/usr/bin/env python3

import argparse
import copy
from datetime import datetime
from distutils.util import strtobool
from distutils.version import LooseVersion
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Optional, List

import torch
from torch.utils import cpp_extension
from torch.testing._internal.common_utils import (
    IS_CI,
    FILE_SCHEMA,
    TEST_WITH_ROCM,
    shell,
    set_cwd,
    parser as common_parser,
)
import torch.distributed as dist
from torch.multiprocessing import Pool, get_context

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))
    from tools.testing.test_selections import (
        get_reordered_tests,
        get_test_case_configs,
    )
    HAVE_TEST_SELECTION_TOOLS = True
except ImportError:
    HAVE_TEST_SELECTION_TOOLS = False
    print(
        "Unable to import test_selections from tools/testing. Running without test selection stats..."
    )

# mac has 3 CPUs and also received the best speedup with 3 processes. Setting this any larger
# will also force use further restrict the amount of memory per process for cuda
NUM_PROCS = 3

def discover_tests(
        base_dir: Optional[pathlib.Path] = None,
        blocklisted_patterns: Optional[List[str]] = None,
        blocklisted_tests: Optional[List[str]] = None,
        extra_tests: Optional[List[str]] = None) -> List[str]:
    """
    Searches for all python files starting with test_ excluding one specified by patterns
    """
    def skip_test_p(name: str) -> bool:
        rc = False
        if blocklisted_patterns is not None:
            rc |= any(name.startswith(pattern) for pattern in blocklisted_patterns)
        if blocklisted_tests is not None:
            rc |= name in blocklisted_tests
        return rc
    cwd = pathlib.Path(__file__).resolve().parent if base_dir is None else base_dir
    all_py_files = list(cwd.glob('**/test_*.py'))
    rc = [str(fname.relative_to(cwd))[:-3] for fname in all_py_files]
    # Invert slashes on Windows
    if sys.platform == "win32":
        rc = [name.replace('\\', '/') for name in rc]
    rc = [test for test in rc if not skip_test_p(test)]
    if extra_tests is not None:
        rc += extra_tests
    return sorted(rc)


TESTS = discover_tests(
    blocklisted_patterns=[
        'ao',
        'bottleneck_test',
        'custom_backend',
        'custom_operator',
        'fx',        # executed by test_fx.py
        'jit',      # executed by test_jit.py
        'mobile',
        'onnx',
        'package',  # executed by test_package.py
        'quantization',  # executed by test_quantization.py
        'autograd',  # executed by test_autograd.py
    ],
    blocklisted_tests=[
        'test_bundled_images',
        'test_cpp_extensions_aot',
        'test_determination',
        'test_jit_fuser',
        'test_jit_simple',
        'test_jit_string',
        'test_kernel_launch_checks',
        'test_metal',
        'test_nnapi',
        'test_segment_reductions',
        'test_static_runtime',
        'test_throughput_benchmark',
        'test_typing',
        "distributed/bin/test_script",
        "distributed/elastic/multiprocessing/bin/test_script",
        "distributed/launcher/bin/test_script",
        "distributed/launcher/bin/test_script_init_method",
        "distributed/launcher/bin/test_script_is_torchelastic_launched",
        "distributed/launcher/bin/test_script_local_rank",
        "distributed/test_c10d_spawn",
        'distributions/test_transforms',
        'distributions/test_utils',
    ],
    extra_tests=[
        "test_cpp_extensions_aot_ninja",
        "test_cpp_extensions_aot_no_ninja",
        "distributed/elastic/timer/api_test",
        "distributed/elastic/timer/local_timer_example",
        "distributed/elastic/timer/local_timer_test",
        "distributed/elastic/events/lib_test",
        "distributed/elastic/metrics/api_test",
        "distributed/elastic/utils/logging_test",
        "distributed/elastic/utils/util_test",
        "distributed/elastic/utils/distributed_test",
        "distributed/elastic/multiprocessing/api_test",
        "doctests",
    ]
)

FSDP_TEST = [test for test in TESTS if test.startswith("distributed/fsdp")]

WINDOWS_BLOCKLIST = [
    "distributed/nn/jit/test_instantiator",
    "distributed/rpc/test_faulty_agent",
    "distributed/rpc/test_tensorpipe_agent",
    "distributed/rpc/test_share_memory",
    "distributed/rpc/cuda/test_tensorpipe_agent",
    "distributed/pipeline/sync/skip/test_api",
    "distributed/pipeline/sync/skip/test_gpipe",
    "distributed/pipeline/sync/skip/test_inspect_skip_layout",
    "distributed/pipeline/sync/skip/test_leak",
    "distributed/pipeline/sync/skip/test_portal",
    "distributed/pipeline/sync/skip/test_stash_pop",
    "distributed/pipeline/sync/skip/test_tracker",
    "distributed/pipeline/sync/skip/test_verify_skippables",
    "distributed/pipeline/sync/test_balance",
    "distributed/pipeline/sync/test_bugs",
    "distributed/pipeline/sync/test_checkpoint",
    "distributed/pipeline/sync/test_copy",
    "distributed/pipeline/sync/test_deferred_batch_norm",
    "distributed/pipeline/sync/test_dependency",
    "distributed/pipeline/sync/test_inplace",
    "distributed/pipeline/sync/test_microbatch",
    "distributed/pipeline/sync/test_phony",
    "distributed/pipeline/sync/test_pipe",
    "distributed/pipeline/sync/test_pipeline",
    "distributed/pipeline/sync/test_stream",
    "distributed/pipeline/sync/test_transparency",
    "distributed/pipeline/sync/test_worker",
    "distributed/elastic/agent/server/test/api_test",
    "distributed/elastic/multiprocessing/api_test",
    "distributed/_shard/checkpoint/test_checkpoint"
    "distributed/_shard/checkpoint/test_file_system_checkpoint"
    "distributed/_shard/sharding_spec/test_sharding_spec",
    "distributed/_shard/sharding_plan/test_sharding_plan",
    "distributed/_shard/sharded_tensor/test_megatron_prototype",
    "distributed/_shard/sharded_tensor/test_sharded_tensor",
    "distributed/_shard/sharded_tensor/test_sharded_tensor_reshard",
    "distributed/_shard/sharded_tensor/ops/test_chunk",
    "distributed/_shard/sharded_tensor/ops/test_elementwise_ops",
    "distributed/_shard/sharded_tensor/ops/test_embedding",
    "distributed/_shard/sharded_tensor/ops/test_embedding_bag",
    "distributed/_shard/sharded_tensor/ops/test_binary_cmp",
    "distributed/_shard/sharded_tensor/ops/test_init",
    "distributed/_shard/sharded_tensor/ops/test_linear",
    "distributed/_shard/sharded_tensor/ops/test_math_ops",
    "distributed/_shard/sharded_tensor/ops/test_matrix_ops",
    "distributed/_shard/sharded_tensor/ops/test_softmax",
    "distributed/_shard/sharded_optim/test_sharded_optim",
    "distributed/_shard/test_partial_tensor",
    "distributed/_shard/test_replicated_tensor",
] + FSDP_TEST

ROCM_BLOCKLIST = [
    "distributed/rpc/test_faulty_agent",
    "distributed/rpc/test_tensorpipe_agent",
    "distributed/rpc/test_share_memory",
    "distributed/rpc/cuda/test_tensorpipe_agent",
    "distributed/_shard/checkpoint/test_checkpoint"
    "distributed/_shard/checkpoint/test_file_system_checkpoint"
    "distributed/_shard/sharding_spec/test_sharding_spec",
    "distributed/_shard/sharding_plan/test_sharding_plan",
    "distributed/_shard/sharded_tensor/test_megatron_prototype",
    "distributed/_shard/sharded_tensor/test_sharded_tensor",
    "distributed/_shard/sharded_tensor/test_sharded_tensor_reshard",
    "distributed/_shard/sharded_tensor/ops/test_chunk",
    "distributed/_shard/sharded_tensor/ops/test_elementwise_ops",
    "distributed/_shard/sharded_tensor/ops/test_embedding",
    "distributed/_shard/sharded_tensor/ops/test_embedding_bag",
    "distributed/_shard/sharded_tensor/ops/test_binary_cmp",
    "distributed/_shard/sharded_tensor/ops/test_init",
    "distributed/_shard/sharded_tensor/ops/test_linear",
    "distributed/_shard/sharded_tensor/ops/test_math_ops",
    "distributed/_shard/sharded_tensor/ops/test_matrix_ops",
    "distributed/_shard/sharded_tensor/ops/test_softmax",
    "distributed/_shard/sharded_optim/test_sharded_optim",
    "distributed/_shard/test_partial_tensor",
    "distributed/_shard/test_replicated_tensor",
    "test_determination",
    "test_jit_legacy",
    "test_openmp",
]

RUN_PARALLEL_BLOCKLIST = [
    "test_cpp_extensions_jit",
    "test_cpp_extensions_open_device_registration",
    "test_jit_disabled",
    "test_mobile_optimizer",
    "test_multiprocessing",
    "test_multiprocessing_spawn",
    "test_namedtuple_return_api",
    "test_overrides",
    "test_show_pickle",
    "test_tensorexpr",
    "test_cuda_primary_ctx",
    "test_cuda_trace",
] + FSDP_TEST

CI_SERIAL_LIST = [
    'test_nn',
    'test_fake_tensor',
    'test_cpp_api_parity',
    'test_reductions',
    'test_cuda',
    'test_jit_cuda_fuser',  # OOM on test_issue_1785, also profiling?
    'test_indexing',
    'test_fx_backends',
    'test_linalg',
    'test_cpp_extensions_jit',
    'test_torch',
    'test_tensor_creation_ops',
    'test_sparse_csr',
    'test_dispatch',
    'nn/test_pooling',
    'distributions/test_distributions',
    'test_autograd',  # slow gradcheck runs a test that checks the cuda memory allocator
    'test_prims',  # slow gradcheck runs a test that checks the cuda memory allocator
]


# A subset of our TEST list that validates PyTorch's ops, modules, and autograd function as expected
CORE_TEST_LIST = [
    "test_autograd",
    "test_modules",
    "test_nn",
    "test_ops",
    "test_ops_gradients",
    "test_ops_jit",
    "test_torch"
]

PYTEST_INCOMPATIBLE = [
    "test_jit",  # test_warn, I think pytest eats warnings
    "test_jit_legacy",  # test_warn, I think pytest eats warnings
    "test_quantization",  # class for op dummy_quant not implemented
    "test_nn",  # test collection results in extra test?
    "test_fx",  # symbolically traced variables cannot be used as inputs to control flow
    "lazy/test_reuse_ir",  # Tried to register multiple backend fallbacks
    "lazy/test_step_closures",  # Tried to register multiple backend fallbacks
    "lazy/test_ts_opinfo",  # Tried to register multiple backend fallbacks
    "test_mps",  # 'TestAvgPool' object has no attribute 'assertRaisesRegex'
    "test_multiprocessing",  # some assertion failure about has_shm_files
    "test_cuda",  # test_mem_get_info, memory freed doesn't match
    "test_autograd",  # memory leak/amount of memory allocated doesn't match
    "doctests",  # doesn't use run_test
]

# if a test file takes longer than 5 min, we add it to TARGET_DET_LIST
SLOW_TEST_THRESHOLD = 300

DISTRIBUTED_TESTS_CONFIG = {}


if dist.is_available():
    DISTRIBUTED_TESTS_CONFIG["test"] = {"WORLD_SIZE": "1"}
    if not TEST_WITH_ROCM and dist.is_mpi_available():
        DISTRIBUTED_TESTS_CONFIG["mpi"] = {
            "WORLD_SIZE": "3",
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-mpi",
        }
    if dist.is_nccl_available():
        DISTRIBUTED_TESTS_CONFIG["nccl"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-nccl",
        }
    if dist.is_gloo_available():
        DISTRIBUTED_TESTS_CONFIG["gloo"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-gloo",
        }
    if dist.is_ucc_available():
        DISTRIBUTED_TESTS_CONFIG["ucc"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-ucc",
            "UCX_TLS": "tcp",
            "UCC_TLS": "nccl,ucp",
            "UCC_TL_UCP_TUNE": "cuda:0",  # don't use UCP TL on CUDA as it is not well supported
        }

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if n.startswith("SIG") and "_" not in n
}

CPP_EXTENSIONS_ERROR = """
Ninja (https://ninja-build.org) is required for some of the C++ extensions
tests, but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`. Alternatively, disable said tests with
`run_test.py --exclude test_cpp_extensions_aot_ninja test_cpp_extensions_jit`.
"""

PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))

JIT_EXECUTOR_TESTS = [
    "test_jit_profiling",
    "test_jit_legacy",
    "test_jit_fuser_legacy",
]

DISTRIBUTED_TESTS = [test for test in TESTS if test.startswith("distributed")]


def discover_functorch_tests():
    pytorch_root = pathlib.Path(__file__).resolve().parent.parent
    functorch_test_dir = os.path.join(pytorch_root, 'functorch', 'test')
    result = discover_tests(pathlib.Path(functorch_test_dir))
    result = [os.path.join(functorch_test_dir, r) for r in result]

    # Sanity check
    assert len(result) >= 8
    return result


FUNCTORCH_TESTS = discover_functorch_tests()

TESTS_REQUIRING_LAPACK = [
    "distributions/test_constraints",
    "distributions/test_distributions",
]


def print_to_stderr(message):
    print(message, file=sys.stderr)


def get_executable_command(options, disable_coverage=False):
    if options.coverage and not disable_coverage:
        executable = ["coverage", "run", "--parallel-mode", "--source=torch"]
    else:
        executable = [sys.executable, "-bb"]
    return executable


def run_test(
    test_module,
    test_directory,
    options,
    launcher_cmd=None,
    extra_unittest_args=None,
    env=None,
) -> int:
    unittest_args = options.additional_unittest_args.copy()
    unittest_args.extend(extra_unittest_args or [])
    if test_module not in PYTEST_INCOMPATIBLE:
        which_shard, num_shards = options.shard or (1, 1)
        subprocess.run(["python", "-m", "pip", "install", "pytest-shard"])
        unittest_args.extend(["--use-pytest", '-vv', '-x', '--reruns=2', '-rfEX',
                              f'--shard-id={which_shard - 1}', f'--num-shards={num_shards}'])

    if IS_CI:
        # use the downloaded test cases configuration, not supported in pytest
        unittest_args.extend(["--import-slow-tests", "--import-disabled-tests"])

    # Extra arguments are not supported with pytest
    executable = get_executable_command(options)

    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test_module + ".py"] + unittest_args

    log_fd, log_path = tempfile.mkstemp(dir=REPO_ROOT / "test" / "test-reports",
                                        prefix=test_module.replace("\\", "-").replace("/", "-"))
    os.close(log_fd)
    command = (launcher_cmd or []) + executable + argv
    print_to_stderr("Executing {} ... [{}]".format(command, datetime.now()))
    with open(log_path, "w") as f:
        ret_code = shell(command, test_directory, stdout=f, stderr=f, env=env)
    print_log_file(test_module, log_path)
    os.remove(log_path)
    return ret_code


def run_test_with_subprocess(test_module, test_directory, options):
    return run_test(test_module, test_directory, options, extra_unittest_args=["--subprocess"])


def _test_cpp_extensions_aot(test_directory, options, use_ninja):
    if use_ninja:
        try:
            cpp_extension.verify_ninja_availability()
        except RuntimeError:
            print(CPP_EXTENSIONS_ERROR)
            return 1

    # Wipe the build folder, if it exists already
    cpp_extensions_test_dir = os.path.join(test_directory, "cpp_extensions")
    cpp_extensions_test_build_dir = os.path.join(cpp_extensions_test_dir, "build")
    if os.path.exists(cpp_extensions_test_build_dir):
        shutil.rmtree(cpp_extensions_test_build_dir)

    # Build the test cpp extensions modules
    shell_env = os.environ.copy()
    shell_env["USE_NINJA"] = str(1 if use_ninja else 0)
    cmd = [sys.executable, "setup.py", "install", "--root", "./install"]
    return_code = shell(cmd, cwd=cpp_extensions_test_dir, env=shell_env)
    if return_code != 0:
        return return_code
    if sys.platform != "win32":
        return_code = shell(
            cmd,
            cwd=os.path.join(cpp_extensions_test_dir, "no_python_abi_suffix_test"),
            env=shell_env,
        )
        if return_code != 0:
            return return_code

    # "install" the test modules and run tests
    python_path = os.environ.get("PYTHONPATH", "")
    from shutil import copyfile

    os.environ['USE_NINJA'] = shell_env['USE_NINJA']
    test_module = "test_cpp_extensions_aot" + ("_ninja" if use_ninja else "_no_ninja")
    copyfile(
        test_directory + "/test_cpp_extensions_aot.py",
        test_directory + "/" + test_module + ".py",
    )
    try:
        cpp_extensions = os.path.join(test_directory, "cpp_extensions")
        install_directory = ""
        # install directory is the one that is named site-packages
        for root, directories, _ in os.walk(os.path.join(cpp_extensions, "install")):
            for directory in directories:
                if "-packages" in directory:
                    install_directory = os.path.join(root, directory)

        assert install_directory, "install_directory must not be empty"
        os.environ["PYTHONPATH"] = os.pathsep.join([install_directory, python_path])
        return run_test(test_module, test_directory, options)
    finally:
        os.environ["PYTHONPATH"] = python_path
        if os.path.exists(test_directory + "/" + test_module + ".py"):
            os.remove(test_directory + "/" + test_module + ".py")
        os.environ.pop('USE_NINJA')


def test_cpp_extensions_aot_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=True)


def test_cpp_extensions_aot_no_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=False)


def test_distributed(test_module, test_directory, options):
    # MPI tests are broken with Python-3.9
    mpi_available = subprocess.call(
        "command -v mpiexec", shell=True
    ) == 0 and sys.version_info < (3, 9)
    if options.verbose and not mpi_available:
        print_to_stderr("MPI not available -- MPI backend tests will be skipped")
    config = DISTRIBUTED_TESTS_CONFIG

    for with_init_file in {True, False}:
        # Run all distributed backends in parallel, trying to run env/file init
        # methods in parallel too ends in failures in which the subprocesses
        # timeout
        pool = Pool(processes=len(config))
        return_codes = []
        tmp_dirs = []

        for backend, env_vars in config.items():
            if sys.platform == "win32" and backend != "gloo":
                continue
            if backend == "mpi" and not mpi_available:
                continue
            if sys.platform == "win32" and not with_init_file:
                continue
            tmp_dir = tempfile.mkdtemp()
            tmp_dirs.append(tmp_dir)
            if options.verbose:
                init_str = "with {} init_method"
                with_init = init_str.format("file" if with_init_file else "env")
                print_to_stderr(
                    "Running distributed tests for the {} backend {}".format(
                        backend, with_init
                    )
                )
            old_environ = dict(os.environ)
            os.environ["TEMP_DIR"] = tmp_dir
            os.environ["BACKEND"] = backend
            os.environ["INIT_METHOD"] = "env://"
            os.environ.update(env_vars)
            if with_init_file:
                if test_module == "test_distributed_spawn":
                    init_method = f"{FILE_SCHEMA}{tmp_dir}/"
                else:
                    init_method = f"{FILE_SCHEMA}{tmp_dir}/shared_init_file"
                os.environ["INIT_METHOD"] = init_method

            try:
                os.mkdir(os.path.join(tmp_dir, "barrier"))
                os.mkdir(os.path.join(tmp_dir, "test_dir"))
                if backend == "mpi":
                    # test mpiexec for --noprefix option
                    with open(os.devnull, "w") as devnull:
                        allowrunasroot_opt = (
                            "--allow-run-as-root"
                            if subprocess.call(
                                'mpiexec --allow-run-as-root -n 1 bash -c ""',
                                shell=True,
                                stdout=devnull,
                                stderr=subprocess.STDOUT,
                            )
                            == 0
                            else ""
                        )
                        noprefix_opt = (
                            "--noprefix"
                            if subprocess.call(
                                f'mpiexec {allowrunasroot_opt} -n 1 --noprefix bash -c ""',
                                shell=True,
                                stdout=devnull,
                                stderr=subprocess.STDOUT,
                            )
                            == 0
                            else ""
                        )

                    mpiexec = ["mpiexec", "-n", "3", noprefix_opt, allowrunasroot_opt]
                    return_code = pool.apply_async(
                        run_test,
                        args=(test_module, test_directory, options),
                        kwds={
                            "launcher_cmd": mpiexec,
                            "env": os.environ.copy(),
                        }
                    )
                else:
                    return_code = pool.apply_async(
                        run_test,
                        args=(test_module, test_directory, options),
                        kwds={
                            "extra_unittest_args": ["--subprocess"],
                            "env": os.environ.copy(),
                        }
                    )

                return_codes.append(return_code)

            finally:
                os.environ.clear()
                os.environ.update(old_environ)

        pool.close()
        # Close the pool and wait for all the processes to finish
        pool.join()

        for tmp_dir in tmp_dirs:
            shutil.rmtree(tmp_dir)

        for return_code in return_codes:
            if return_code.get() != 0:
                return return_code

    return 0


def run_doctests(test_module, test_directory, options):
    """
    Assumes the incoming test module is called doctest, and simply executes the
    xdoctest runner on the torch library itself.
    """
    import xdoctest
    import pathlib
    pkgpath = pathlib.Path(torch.__file__).parent

    #
    enabled = {
        # TODO: expose these options to the user
        # Temporary disable all feature-conditional tests
        # 'lapack': 'auto',
        # 'cuda': 'auto',
        # 'cuda1': 'auto',
        # 'qengine': 'auto',
        'lapack': 0,
        'cuda': 0,
        'cuda1': 0,
        'qengine': 0,
    }

    # Resolve "auto" based on a test to determine if the feature is available.
    if enabled['cuda'] == 'auto' and torch.cuda.is_available():
        enabled['cuda'] = True

    if enabled['cuda1'] == 'auto' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        enabled['cuda1'] = True

    if enabled['lapack'] == 'auto' and torch._C.has_lapack:
        enabled['lapack'] = True

    if enabled['qengine'] == 'auto':
        try:
            # Is there a better check if quantization is enabled?
            import torch.nn.quantized as nnq  # NOQA
            torch.backends.quantized.engine = 'qnnpack'
            torch.backends.quantized.engine = 'fbgemm'
        except (ImportError, RuntimeError):
            ...
        else:
            enabled['qengine'] = True

    # Set doctest environment variables
    if enabled['cuda']:
        os.environ['TORCH_DOCTEST_CUDA'] = '1'

    if enabled['cuda1']:
        os.environ['TORCH_DOCTEST_CUDA1'] = '1'

    if enabled['lapack']:
        os.environ['TORCH_DOCTEST_LAPACK'] = '1'

    if enabled['qengine']:
        os.environ['TORCH_DOCTEST_QENGINE'] = '1'

    pkgpath = os.path.dirname(torch.__file__)
    xdoctest_config = {
        'global_exec': r'\n'.join([
            'from torch import nn',
            'import torch.nn.functional as F',
            'import torch',
        ]),
        'style': 'google',
        'options': '+IGNORE_WHITESPACE',
    }
    xdoctest_verbose = max(1, options.verbose)
    run_summary = xdoctest.runner.doctest_module(
        os.fspath(pkgpath), config=xdoctest_config, verbose=xdoctest_verbose,
        command=options.xdoctest_command, argv=[])
    result = 1 if run_summary.get('n_failed', 0) else 0
    return result


def print_log_file(test: str, file_path: str) -> None:
    with open(file_path, "r") as f:
        print_to_stderr("")
        print_to_stderr(f"PRINT LOG FILE of {test} ({file_path})")
        print_to_stderr(f"##[group]PRINT LOG FILE of {test} ({file_path})")
        print_to_stderr(f.read())
        print_to_stderr("##[endgroup]")
        print_to_stderr(f"FINISHED PRINT LOG FILE of {test} ({file_path})")
        print_to_stderr("")


def run_test_ops(test_module, test_directory, options):
    if 'slow-gradcheck' in os.getenv("BUILD_ENVIRONMENT", ""):
        # there are a lot of tests that take up a lot of space in slowgrad check, so don't bother parallelizing
        # it's also on periodic so we don't care about TTS as much
        return run_test(test_module, test_directory, copy.deepcopy(options),
                        extra_unittest_args=["--use-pytest", '-vv', '-x', '--reruns=2', '-rfEX'],
                        )

    return_codes = []
    os.environ["PARALLEL_TESTING"] = "1"
    pool = Pool(NUM_PROCS)
    for i in range(NUM_PROCS):
        return_code = pool.apply_async(run_test, args=(test_module, test_directory, copy.deepcopy(options)),
                                       kwds={"extra_unittest_args": ["--use-pytest", '-vv', '-x', '--reruns=2', '-rfEX',
                                                                     f'--shard-id={i}', f'--num-shards={NUM_PROCS}',
                                                                     "-k=not _linalg_cholesky_"],
                                             })
        return_codes.append(return_code)
    pool.close()
    pool.join()
    del os.environ['PARALLEL_TESTING']

    for return_code in return_codes:
        if return_code.get() != 0:
            return return_code.get()
    return_code = run_test(test_module, test_directory, copy.deepcopy(options),
                           extra_unittest_args=["--use-pytest", '-vv', '-x', '--reruns=2', '-rfEX',
                                                "-k=_linalg_cholesky_"],
                           )
    return return_code


CUSTOM_HANDLERS = {
    "test_cuda_primary_ctx": run_test_with_subprocess,
    "test_cuda_trace": run_test_with_subprocess,
    "test_cpp_extensions_aot_no_ninja": test_cpp_extensions_aot_no_ninja,
    "test_cpp_extensions_aot_ninja": test_cpp_extensions_aot_ninja,
    "distributed/test_distributed_spawn": test_distributed,
    "distributed/algorithms/quantization/test_quantization": test_distributed,
    "distributed/test_c10d_nccl": run_test_with_subprocess,
    "distributed/test_c10d_gloo": run_test_with_subprocess,
    "distributed/test_c10d_common": run_test_with_subprocess,
    "distributed/test_c10d_spawn_gloo": run_test_with_subprocess,
    "distributed/test_c10d_spawn_nccl": run_test_with_subprocess,
    "distributed/test_store": run_test_with_subprocess,
    "distributed/test_pg_wrapper": run_test_with_subprocess,
    "distributed/rpc/test_faulty_agent": run_test_with_subprocess,
    "distributed/rpc/test_tensorpipe_agent": run_test_with_subprocess,
    "distributed/rpc/test_share_memory": run_test_with_subprocess,
    "distributed/rpc/cuda/test_tensorpipe_agent": run_test_with_subprocess,
    "doctests": run_doctests,
    "test_ops": run_test_ops,
    "test_ops_gradients": run_test_ops,
    "test_ops_jit": run_test_ops,
}


def parse_test_module(test):
    return test.split(".")[0]


class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super(TestChoices, self).__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PyTorch unit test suite",
        epilog="where TESTS is any of: {}".format(", ".join(TESTS)),
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[common_parser]
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose information and test-by-test results",
    )
    parser.add_argument("--jit", "--jit", action="store_true", help="run all jit tests")
    parser.add_argument(
        "--distributed-tests",
        "--distributed-tests",
        action="store_true",
        help="run all distributed tests",
    )
    parser.add_argument(
        "--functorch",
        "--functorch",
        action="store_true",
        help=(
            "If this flag is present, we will only run functorch tests. "
            "If this flag is not present, we will not run any functorch tests. "
            "This requires functorch to already be installed."
        )
    )
    parser.add_argument(
        "-core",
        "--core",
        action="store_true",
        help="Only run core tests, or tests that validate PyTorch's ops, modules,"
        "and autograd. They are defined by CORE_TEST_LIST."
    )
    parser.add_argument(
        "-pt",
        "--pytest",
        action="store_true",
        help="If true, use `pytest` to execute the tests. E.g., this runs "
        "TestTorch with pytest in verbose and coverage mode: "
        "python run_test.py -vci torch -pt",
    )
    parser.add_argument(
        "-c",
        "--coverage",
        action="store_true",
        help="enable coverage",
        default=PYTORCH_COLLECT_COVERAGE,
    )
    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar="TESTS",
        help="select a set of tests to include (defaults to ALL tests)."
        " tests must be a part of the TESTS list defined in run_test.py",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        choices=TESTS,
        metavar="TESTS",
        default=[],
        help="select a set of tests to exclude",
    )
    parser.add_argument(
        "-f",
        "--first",
        choices=TESTS,
        metavar="TESTS",
        help="select the test to start from (excludes previous tests)",
    )
    parser.add_argument(
        "-l",
        "--last",
        choices=TESTS,
        metavar="TESTS",
        help="select the last test to run (excludes following tests)",
    )
    parser.add_argument(
        "--bring-to-front",
        nargs="+",
        choices=TestChoices(TESTS),
        default=[],
        metavar="TESTS",
        help="select a set of tests to run first. This can be used in situations"
        " where you want to run all tests, but care more about some set, "
        "e.g. after making a change to a specific component",
    )
    parser.add_argument(
        "--ignore-win-blocklist",
        action="store_true",
        help="always run blocklisted windows tests",
    )
    # NS: Disable target determination until it can be made more reliable
    # parser.add_argument(
    #     "--determine-from",
    #     help="File of affected source filenames to determine which tests to run.",
    # )
    parser.add_argument(
        "--continue-through-error",
        action="store_true",
        help="Runs the full test suite despite one of the tests failing",
        default=strtobool(os.environ.get("CONTINUE_THROUGH_ERROR", "False")),
    )
    parser.add_argument(
        "additional_unittest_args",
        nargs="*",
        help="additional arguments passed through to unittest, e.g., "
        "python run_test.py -i sparse -- TestSparse.test_factory_size_check",
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        help="runs a shard of the tests (taking into account other selections), e.g., "
        "--shard 2 3 will break up the selected tests into 3 shards and run the tests "
        "in the 2nd shard (the first number should not exceed the second)",
    )
    parser.add_argument(
        "--exclude-jit-executor",
        action="store_true",
        help="exclude tests that are run for a specific jit config",
    )
    parser.add_argument(
        "--exclude-distributed-tests",
        action="store_true",
        help="exclude distributed tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the test that will run.",
    )
    parser.add_argument(
        "--xdoctest-command",
        default='list',
        help=(
            "Control the specific doctest action. "
            "Use 'list' to simply parse doctests and check syntax. "
            "Use 'all' to execute all doctests or specify a specific "
            "doctest to run")
    )
    return parser.parse_args()


def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of selected tests.

    This function is used to determine the indices when slicing the list of selected tests when
    ``options.first``(:attr:`find_last_index`=False) and/or ``options.last``(:attr:`find_last_index`=True) are used.

    :attr:`selected_tests` can be a list that contains multiple consequent occurrences of tests
    as part of the same test module, e.g.:

    ```
    selected_tests = ['autograd', 'cuda', **'torch.TestTorch.test_acos',
                     'torch.TestTorch.test_tan', 'torch.TestTorch.test_add'**, 'utils']
    ```

    If :attr:`test`='torch' and :attr:`find_last_index`=False, result should be **2**.
    If :attr:`test`='torch' and :attr:`find_last_index`=True, result should be **4**.

    Args:
        test (str): Name of test to lookup
        selected_tests (list): List of tests
        find_last_index (bool, optional): should we lookup the index of first or last
            occurrence (first is default)

    Returns:
        index of the first or last occurrence of the given test
    """
    idx = 0
    found_idx = -1
    for t in selected_tests:
        if t.startswith(test):
            found_idx = idx
            if not find_last_index:
                break
        idx += 1
    return found_idx


def exclude_tests(exclude_list, selected_tests, exclude_message=None):
    for exclude_test in exclude_list:
        tests_copy = selected_tests[:]
        for test in tests_copy:
            if test.startswith(exclude_test):
                if exclude_message is not None:
                    print_to_stderr("Excluding {} {}".format(test, exclude_message))
                selected_tests.remove(test)
    return selected_tests


def must_serial(file: str) -> bool:
    return (
        "distributed" in os.getenv("TEST_CONFIG", "") or
        "functorch" in os.getenv("TEST_CONFIG", "") or
        "dynamo" in os.getenv("TEST_CONFIG", "") or
        "distributed" in file or
        file in CUSTOM_HANDLERS or
        file in RUN_PARALLEL_BLOCKLIST or
        file in CI_SERIAL_LIST
    )


def get_selected_tests(options):
    selected_tests = options.include

    # filter if there's JIT only and distributed only test options
    if options.jit:
        selected_tests = list(
            filter(lambda test_name: "jit" in test_name, selected_tests)
        )

    if options.distributed_tests:
        selected_tests = list(
            filter(lambda test_name: test_name in DISTRIBUTED_TESTS, selected_tests)
        )

    # Filter to only run core tests when --core option is specified
    if options.core:
        selected_tests = list(
            filter(lambda test_name: test_name in CORE_TEST_LIST, selected_tests)
        )

    if options.functorch:
        selected_tests = FUNCTORCH_TESTS

    # process reordering
    if options.bring_to_front:
        to_front = set(options.bring_to_front)
        selected_tests = options.bring_to_front + list(
            filter(lambda name: name not in to_front, selected_tests)
        )

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[: last_index + 1]

    # process exclusion
    if options.exclude_jit_executor:
        options.exclude.extend(JIT_EXECUTOR_TESTS)

    if options.exclude_distributed_tests:
        options.exclude.extend(DISTRIBUTED_TESTS)

    # these tests failing in CUDA 11.6 temporary disabling. issue https://github.com/pytorch/pytorch/issues/75375
    if torch.version.cuda is not None and LooseVersion(torch.version.cuda) >= "11.6":
        options.exclude.extend(["distributions/test_constraints"])

    selected_tests = exclude_tests(options.exclude, selected_tests)

    if sys.platform == "win32" and not options.ignore_win_blocklist:
        target_arch = os.environ.get("VSCMD_ARG_TGT_ARCH")
        if target_arch != "x64":
            WINDOWS_BLOCKLIST.append("cpp_extensions_aot_no_ninja")
            WINDOWS_BLOCKLIST.append("cpp_extensions_aot_ninja")
            WINDOWS_BLOCKLIST.append("cpp_extensions_jit")
            WINDOWS_BLOCKLIST.append("jit")
            WINDOWS_BLOCKLIST.append("jit_fuser")

        # This is exception that's caused by this issue https://github.com/pytorch/pytorch/issues/69460
        # This below code should be removed once this issue is solved
        if torch.version.cuda is not None and LooseVersion(torch.version.cuda) >= "11.5":
            WINDOWS_BLOCKLIST.append("test_cpp_extensions_aot")
            WINDOWS_BLOCKLIST.append("test_cpp_extensions_aot_ninja")
            WINDOWS_BLOCKLIST.append("test_cpp_extensions_aot_no_ninja")

        selected_tests = exclude_tests(WINDOWS_BLOCKLIST, selected_tests, "on Windows")

    elif TEST_WITH_ROCM:
        selected_tests = exclude_tests(ROCM_BLOCKLIST, selected_tests, "on ROCm")

    # sharding
    if options.shard:
        assert len(options.shard) == 2, "Unexpected shard format"
        assert min(options.shard) > 0, "Shards must be positive numbers"
        which_shard, num_shards = options.shard
        assert which_shard <= num_shards, "Selected shard must be less than or equal to total number of shards"
        assert num_shards <= len(selected_tests), f"Number of shards must be less than {len(selected_tests)}"
        if which_shard != 1:
            selected_tests = [x for x in selected_tests if x not in PYTEST_INCOMPATIBLE]

    # skip all distributed tests if distributed package is not available.
    if not dist.is_available():
        selected_tests = exclude_tests(DISTRIBUTED_TESTS, selected_tests,
                                       "PyTorch is built without distributed support.")

    # skip tests that require LAPACK when it's not available
    if not torch._C.has_lapack:
        selected_tests = exclude_tests(TESTS_REQUIRING_LAPACK, selected_tests,
                                       "PyTorch is built without LAPACK support.")

    return selected_tests


def run_test_module(test: str, test_directory: str, options) -> Optional[str]:
    test_module = parse_test_module(test)

    # Printing the date here can help diagnose which tests are slow
    print_to_stderr("Running {} ... [{}]".format(test, datetime.now()))
    handler = CUSTOM_HANDLERS.get(test_module, run_test)
    return_code = handler(test_module, test_directory, options)
    assert isinstance(return_code, int) and not isinstance(return_code, bool), (
        f"While running {test} got non integer return code {return_code}"
    )
    if return_code == 0:
        return None

    message = f"{test} failed!"
    if return_code < 0:
        # subprocess.Popen returns the child process' exit signal as
        # return code -N, where N is the signal number.
        signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
        message += f" Received signal: {signal_name}"
    return message


def main():
    options = parse_args()

    test_directory = str(REPO_ROOT / "test")
    selected_tests = get_selected_tests(options)

    if options.verbose:
        print_to_stderr("Selected tests:\n {}".format("\n ".join(selected_tests)))

    if options.dry_run:
        return

    if options.coverage and not PYTORCH_COLLECT_COVERAGE:
        shell(["coverage", "erase"])

    if IS_CI and HAVE_TEST_SELECTION_TOOLS:
        selected_tests = get_reordered_tests(selected_tests)
        # downloading test cases configuration to local environment
        get_test_case_configs(dirpath=test_directory)
    failure_messages = []

    selected_tests_parallel = [x for x in selected_tests if not must_serial(x)]
    selected_tests_serial = [x for x in selected_tests if x not in selected_tests_parallel]
    print_to_stderr("parallel tests:\n {}".format("\n ".join(selected_tests_parallel)))
    print_to_stderr("serial tests:\n {}".format("\n ".join(selected_tests_serial)))

    pool = get_context("spawn").Pool(NUM_PROCS, maxtasksperchild=1)
    os.makedirs(REPO_ROOT / "test" / "test-reports", exist_ok=True)

    def success_callback(err_message):
        if err_message is None:
            return True
        failure_messages.append(err_message)
        print_to_stderr(err_message)
        if not options.continue_through_error:
            pool.terminate()
        return False

    try:
        os.environ['PARALLEL_TESTING'] = '1'
        for test in selected_tests_parallel:
            pool.apply_async(run_test_module, args=(test, test_directory,
                             copy.deepcopy(options)), callback=success_callback)
        pool.close()
        pool.join()
        del os.environ['PARALLEL_TESTING']

        if not options.continue_through_error and len(failure_messages) != 0:
            raise RuntimeError("\n".join(failure_messages))

        for test in selected_tests_serial:
            options_clone = copy.deepcopy(options)
            err_message = run_test_module(test, test_directory, options_clone)
            if err_message is None:
                continue
            failure_messages.append(err_message)
            if not options_clone.continue_through_error:
                raise RuntimeError(err_message)
            print_to_stderr(err_message)
    finally:
        pool.terminate()
        pool.join()

        if options.coverage:
            from coverage import Coverage

            with set_cwd(test_directory):
                cov = Coverage()
                if PYTORCH_COLLECT_COVERAGE:
                    cov.load()
                cov.combine(strict=False)
                cov.save()
                if not PYTORCH_COLLECT_COVERAGE:
                    cov.html_report()

    if len(failure_messages) != 0:
        for err in failure_messages:
            print_to_stderr(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
