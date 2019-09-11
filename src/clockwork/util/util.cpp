#include "clockwork/util/util.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <string.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <stdio.h>


#include "utils.h"

namespace clockwork {
namespace util {	

std::uint64_t now() {
	auto t = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

std::string nowString() {
  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8>>::type> Days;

  Days days = std::chrono::duration_cast<Days>(duration);                               duration -= days;
  auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);                duration -= hours;
  auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);            duration -= minutes;
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);            duration -= seconds;
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);  duration -= milliseconds;
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);  duration -= microseconds;
  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

  std::stringstream ss;
  ss << hours.count() << ":" << minutes.count() << ":" << seconds.count() << "." << milliseconds.count() << " " << microseconds.count() << " " << nanoseconds.count();
  return ss.str();
}


void set_core(unsigned core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
  }
}

void setCudaFlags() {
  cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

std::string getGPUmodel(int deviceNumber) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  if (deviceNumber < 0 || deviceNumber >= nDevices) {
    std::cout << nDevices << " devices found, invalid deviceNumber " << deviceNumber << std::endl;
    exit(-1);
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceNumber);
  return std::string(prop.name);
}


char* getGPUModelToBuffer(int deviceNumber, char* buf) {
  strcpy(buf, getGPUmodel(deviceNumber).c_str());
  return buf;
}

}
}