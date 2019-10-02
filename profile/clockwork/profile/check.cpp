#include <sys/time.h>
#include <sys/resource.h>
#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/so.h"
#include "clockwork/model/cuda.h"
#include <nvml.h>

using namespace clockwork;


void check_environment() {
    bool environmentIsOK = true;
    if (!util::is_cuda_cache_disabled()) {
        std::cout << "✘ CUDA cache is enabled!  It should be disabled by setting environment variable CUDA_CACHE_DISABLE=1" << std::endl;
        environmentIsOK = false;
    } else {
        std::cout << "✔ CUDA cache is disabled" << std::endl;
    }
    if (util::is_force_ptx_jit_enabled()) {
        std::cout << "✘ PTX JIT is being forced!  Unset the CUDA_FORCE_PTX_JIT environment variable" << std::endl;
        environmentIsOK = false;
    } else {
        std::cout << "✔ PTX JIT is not forced" << std::endl;
    }

    struct rlimit rlim;
    getrlimit(RLIMIT_NOFILE, &rlim);
    if (rlim.rlim_cur < 65535) {
        std::cout << "✘ Resource limit on number of open files (RLIMIT_NOFILE) is " << rlim.rlim_cur << ", require at least 65535" << std::endl;
        environmentIsOK = false;
    } else {
        std::cout << "✔ RLIMIT_NOFILE is " << rlim.rlim_cur << std::endl;
    }

    getrlimit(RLIMIT_MEMLOCK, &rlim);
    if (rlim.rlim_cur < 1024L * 1024L * 1024L * 1024L) {
        std::cout << "✘ Resource limit on memlocked pages is " << rlim.rlim_cur << ", require unlimited" << std::endl;
        environmentIsOK = false;
    } else {
        std::cout << "✔ RLIMIT_MEMLOCK is " << rlim.rlim_cur << std::endl;
    }

    unsigned num_gpus = util::get_num_gpus();
    std::cout << "Found " << num_gpus << " GPUs" << std::endl;
    for (unsigned i = 0; i < num_gpus; i++) {

        if (!util::is_gpu_exclusive(i)) { // TODO: check all GPUs
            std::cout << "  ✘ GPU " << i << " is not in exclusive mode; set with `nvidia-smi -i " << i << " -c 3` or set for all GPUs with `nvndia-smi -c 3`" << std::endl;
        } else {
            std::cout << "  ✔ GPU " << i << " is in exclusive mode" << std::endl;
        }


    }

    util::nvml();

    REQUIRE(environmentIsOK);
}

TEST_CASE("Check environment variables", "[profile] [check]") {
    check_environment();
}

void print_system_status() {
    unsigned num_gpus = util::get_num_gpus();
    std::cout << num_gpus << " attached GPUs" << std::endl;

    std::cout << "GPU compute capability:" << std::endl;
    for (unsigned i = 0; i < num_gpus; i++) {
        std::pair<int, int> v = util::get_compute_capability(i);
        std::cout << "  GPU " << i << " = " << v.first << ", " << v.second << std::endl;
    }

    std::cout << "GPU core affinity:" << std::endl;
    for (unsigned i = 0; i < num_gpus; i++) {
        std::vector<unsigned> cores = util::get_gpu_core_affinity(i);
        std::cout << "  GPU " << i << " =";
        for (unsigned &core : cores) {
            std::cout << " " << core;
        }
        std::cout << std::endl;
    }





}


TEST_CASE("Print system status", "[profile] [status]") {
    print_system_status();
}
