#include <sys/time.h>
#include <sys/resource.h>
#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/so.h"
#include "clockwork/model/cuda.h"

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

    REQUIRE(environmentIsOK);
}

TEST_CASE("Check environment variables", "[profile] [check]") {
    check_environment();
}