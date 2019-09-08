#ifndef _CLOCKWORK_UTIL_H_
#define _CLOCKWORK_UTIL_H_

#include <cstdint>


namespace clockwork {
namespace util {

// High-resolution timer, current time in nanoseconds
std::uint64_t now();

}
}


#endif