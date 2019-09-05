#ifndef _CLOCKWORK_UTIL_H_
#define _CLOCKWORK_UTIL_H_

#include <chrono>

namespace clockwork {
namespace util {

std::uint64_t now() {
	auto t = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

}
}


#endif