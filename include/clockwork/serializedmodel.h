
#ifndef TVM_CLOCKWORK_SERIALIZED_MODEL_H_
#define TVM_CLOCKWORK_SERIALIZED_MODEL_H_

#include <dlfcn.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>

namespace clockwork {
namespace binary {

struct DLTensorDef {
	uint64_t offset;
	std::vector<int64_t> shape;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(offset),
        PODS_MDR(shape)
    )
};

struct Op {
	std::vector<DLTensorDef> inputs;
	unsigned so_function;
	std::vector<unsigned> cuda_functions;
	std::vector<uint64_t> workspace_allocs;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(inputs),
        PODS_MDR(so_function),
        PODS_MDR(cuda_functions),
        PODS_MDR(workspace_allocs)
    )
};

struct MinModel {
	uint64_t total_memory;
	uint64_t weights_memory;
	std::vector<std::string> so_functions;
	std::vector<std::string> cuda_functions;
	std::vector<Op> ops;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(total_memory),
        PODS_MDR(weights_memory),
        PODS_MDR(so_functions),
        PODS_MDR(cuda_functions),
        PODS_MDR(ops)
    )
};


  // // Library handle
  // void* lib_handle_{nullptr};
  // // load the library
  // void Load(const std::string& name) {
  //   lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
  //   CHECK(lib_handle_ != nullptr)
  //       << "Failed to load dynamic shared library " << name
  //       << " " << dlerror();
  //   // std::cout << "Loaded dynamic shared library " << name << std::endl;
  // }
  // void* GetSymbol(const char* name) {
  //   return dlsym(lib_handle_, name);
  // }
  // void Unload() {
  //   dlclose(lib_handle_);
  //   // std::cout << "Unload dynamic shared library " << std::endl;
  // }



}
}

#endif