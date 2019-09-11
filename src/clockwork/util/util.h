#ifndef _CLOCKWORK_UTIL_H_
#define _CLOCKWORK_UTIL_H_

#include <cstdint>
#include <string>


namespace clockwork {
namespace util {

// High-resolution timer, current time in nanoseconds
std::uint64_t now();

std::string nowString();

void set_core(unsigned core);

void setCudaFlags();

std::string getGPUmodel(int deviceNumber);

extern "C" char* getGPUModelToBuffer(int deviceNumber, char* buf);

}
}


#endif