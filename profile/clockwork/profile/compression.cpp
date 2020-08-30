#include <sys/time.h>
#include <sys/resource.h>
#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include "clockwork/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/so.h"
#include "clockwork/model/cuda.h"
#include "clockwork/thread.h"
#include <nvml.h>
#include <lz4.h>

using namespace clockwork;

TEST_CASE("Profile compression throughput & latency", "[profile] [compression]") {
    util::InputGenerator* generator = new util::InputGenerator();

    for (unsigned i = 0; i < 10; i++) {
        size_t size = 602112;
        char* buf;
        generator->generateInput(size, &buf);

        std::vector<uint8_t> bytes(buf, buf+size);

        const int max_dst_size = LZ4_compressBound(size);
        char* compressed = static_cast<char*>(malloc((size_t) max_dst_size));
        const int compressed_size = LZ4_compress_default(buf, compressed, size, max_dst_size);

        std::cout << size << " reduced to " << compressed_size << std::endl;
    }

    for (unsigned i = 0; i < 10; i++) {
        size_t size = 602112;
        char* buf;
        size_t compressed_size;
        generator->generatePrecompressedInput(size, &buf, &compressed_size);

        std::cout << "precompressed " << size << " reduced to " << compressed_size << std::endl;
    }
}