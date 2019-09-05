#include "clockwork/util.h"
#include <chrono>

namespace clockwork {
namespace util {	

std::uint64_t now() {
	auto t = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

}
}