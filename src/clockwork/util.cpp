#include "clockwork/util.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <string.h>
#include <pthread.h>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <istream>
#include "clockwork/cuda_common.h"
#include <cuda_runtime.h>
#include <dmlc/logging.h>
#include <nvml.h>
#include <boost/filesystem.hpp>
#include <sys/stat.h>
#include <libgen.h>


namespace clockwork {
namespace util {	

std::uint64_t now() {
  return nanos(hrt());
}

std::chrono::high_resolution_clock::time_point hrt() {
  return std::chrono::high_resolution_clock::now();
}

std::uint64_t nanos(std::chrono::high_resolution_clock::time_point t) {
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

unsigned get_num_cores() {
  return std::thread::hardware_concurrency();
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

unsigned get_num_gpus() {
  nvmlReturn_t status;

  status = nvmlInit();
  CHECK(status == NVML_SUCCESS);

  unsigned deviceCount;
  status = nvmlDeviceGetCount(&deviceCount);
  CHECK(status == NVML_SUCCESS);

  status = nvmlShutdown();
  CHECK(status == NVML_SUCCESS);

  return deviceCount;
}

std::vector<unsigned> get_gpu_core_affinity(unsigned deviceId) {

  unsigned len = (get_num_cores() + 63) / 64;

  std::vector<uint64_t> bitmaps(len);

  nvmlReturn_t status;

  status = nvmlInit();
  CHECK(status == NVML_SUCCESS);

  nvmlDevice_t device;
  status = nvmlDeviceGetHandleByIndex(deviceId, &device);
  CHECK(status == NVML_SUCCESS);

  // Fill bitmaps with the ideal CPU affinity for the device
  // (see https://helpmanual.io/man3/nvmlDeviceGetCpuAffinity/)
  status = nvmlDeviceGetCpuAffinity(device, bitmaps.size(), bitmaps.data());
  CHECK(status == NVML_SUCCESS);

  std::vector<unsigned> cores;

  unsigned core = 0;
  for (unsigned i = 0; i < bitmaps.size(); i++) {
    for (unsigned j = 0; j < 64; j++) {
      if (((bitmaps[i] >> j) & 0x01) == 0x01) {
        cores.push_back(core);
      }
      core++;
    }
  }

  status = nvmlShutdown();
  CHECK(status == NVML_SUCCESS);

  return cores;
}

void setCudaFlags() {
  cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
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

void setCurrentThreadMaxPriority() {
  pthread_t thId = pthread_self();
  
  struct sched_param params;
  params.sched_priority = sched_get_priority_max(SCHED_FIFO);
  int ret = pthread_setschedparam(thId, SCHED_FIFO, &params);
  if (ret != 0) {
    std::cout << "Warning!  Cannot set thread priority.  Don't forget to set rtprio unlimited in limits.conf.  See README for details." << std::endl;
    return;
  }

  int policy = 0;
  ret = pthread_getschedparam(thId, &policy, &params);
  CHECK(ret == 0) << "Could not retrieve thread scheduler parameters for setting thread priority";
  CHECK(policy == SCHED_FIFO) << "Inconsistent thread scheduler parameters encountered";
}


void readFileAsString(const std::string &filename, std::string &dst) {
  std::ifstream in(filename, std::ios::binary);
  dst = std::string(
      std::istreambuf_iterator<char>(in), 
      std::istreambuf_iterator<char>());
  in.close();
}

struct path_leaf_string
{
    std::string operator()(const boost::filesystem::directory_entry& entry) const
    {
        return entry.path().leaf().string();
    }
};

std::vector<std::string> listdir(std::string directory) {
  std::vector<std::string> filenames;
  boost::filesystem::path p(directory);
  boost::filesystem::directory_iterator start(p);
  boost::filesystem::directory_iterator end;
  std::transform(start, end, std::back_inserter(filenames), path_leaf_string());
  return filenames;
}

bool exists(std::string filename) {
  struct stat buffer;   
  return (stat (filename.c_str(), &buffer) == 0); 
}

long filesize(std::string filename) {
    struct stat buffer;
    int rc = stat(filename.c_str(), &buffer);
    return rc == 0 ? buffer.st_size : -1;
}

thread_local cudaStream_t current_stream;

void initializeCudaStream(unsigned gpu_id, int priority) {
  CUDA_CALL(cudaSetDevice(gpu_id));
  cudaStream_t stream;  
  CUDA_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
  SetStream(stream);
}

void SetStream(cudaStream_t stream) {
  current_stream = stream;
}

cudaStream_t Stream() {
  return current_stream;
}

std::string get_clockwork_directory()
{
	int bufsize = 1024;
	char buf[bufsize];
	int len = readlink("/proc/self/exe", buf, bufsize);
	return dirname(dirname(buf));
}

std::string get_example_model_path(std::string model_name)
{
  return get_example_model_path(get_clockwork_directory(), model_name);
}


std::string get_example_model_path(std::string clockwork_directory, std::string model_name) {
  return clockwork_directory + "/resources/" + model_name + "/model";
}

}
}
