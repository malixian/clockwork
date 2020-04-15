/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */
#include "clockwork-convert/tvm/decoupled_graph_runtime.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>

namespace tvm {
namespace runtime {

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOG_PREFIX() now() << " " << __FILENAME__ << ":" << __LINE__ << " \t"

/*!
 * \brief Run all the operations one by one.
 */
void DecoupledGraphRuntime::Run() {
  // setup the array and requirements.
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
  }
}

std::vector<std::vector<WorkspaceAlloc>>* DecoupledGraphRuntime::ExtractWorkspaceAllocs() {
  std::vector<std::vector<WorkspaceAlloc>>* allocs = new std::vector<std::vector<WorkspaceAlloc>>(op_execs_.size());
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    ManagedCUDADeviceAPI::Global()->tracker.enabled = true;
    if (op_execs_[i]) op_execs_[i]();
    (*allocs)[i] = ManagedCUDADeviceAPI::Global()->tracker.get();
    ManagedCUDADeviceAPI::Global()->tracker.clear();
    ManagedCUDADeviceAPI::Global()->tracker.enabled = false;
  }
  return allocs;
}

/*!
 * \brief Initialize the graph executor with graph and context.
 * \param graph_json The execution graph.
 * \param module The module containing the compiled functions for the host
 * processor.
 * \param ctxs The context of the host and devices where graph nodes will be
 * executed on.
 */
void DecoupledGraphRuntime::Init(const std::string& graph_json,
                        tvm::runtime::Module module,
                        const std::vector<TVMContext>& ctxs,
                        bool contiguous) {
#ifndef _LIBCPP_SGX_NO_IOSTREAMS
  std::istringstream is(graph_json);
#else
  std::string is = graph_json;
#endif
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  module_ = module;
  ctxs_ = ctxs;
  if (contiguous) this->SetupStorageContiguous();
  else this->SetupStorage();
  //this->SetupOpExecs();
}
/*!
 * \brief Get the input index given the name of input.
 * \param name The name of the input.
 * \return The index of input.
 */
int DecoupledGraphRuntime::GetInputIndex(const std::string& name) {
  for (size_t i = 0; i< input_nodes_.size(); ++i) {
    uint32_t nid = input_nodes_[i];
    if (nodes_[nid].name == name) {
      return static_cast<int>(i);
    }
  }
  LOG(WARNING) << "Warning: cannot find \"" << name << "\" among input";
  return -1;
}
/*!
 * \brief set index-th input to the graph.
 * \param index The input index.
 * \param data_in The input data.
 */
void DecoupledGraphRuntime::SetInput(int index, DLTensor* data_in) {
  CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  data_entry_[eid].CopyFrom(data_in);
}
/*!
 * \brief Get the number of outputs
 *
 * \return The number of outputs from graph.
 */
int DecoupledGraphRuntime::NumOutputs() const {
  return outputs_.size();
}
/*!
 * \brief Return NDArray for given input index.
 * \param index The input index.
 *
 * \return NDArray corresponding to given input node index.
 */
NDArray DecoupledGraphRuntime::GetInput(int index) const {
  CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  return data_entry_[eid];
}
/*!
 * \brief Return NDArray for given output index.
 * \param index The output index.
 *
 * \return NDArray corresponding to given output node index.
 */
NDArray DecoupledGraphRuntime::GetOutput(int index) const {
  CHECK_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);
  return data_entry_[eid];
}
/*!
 * \brief Copy index-th output to data_out.
 * \param index The output index.
 * \param data_out the output data.
 */
void DecoupledGraphRuntime::CopyOutputTo(int index, DLTensor* data_out) {
  CHECK_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);

  // Check the shapes to avoid receiving in different dimension but same size.
  const NDArray& data = data_entry_[eid];
  CHECK_EQ(data->ndim, data_out->ndim);
  for (int32_t j = 0; j < data->ndim; ++j) {
    CHECK_EQ(data->shape[j], data_out->shape[j]);
  }

  data_entry_[eid].CopyTo(data_out);
}

/*!
 * \brief Load parameters from parameter blob to cpu memory.
 * \param param_blob A binary blob of parameter.
 */
void DecoupledGraphRuntime::LoadParams(const std::string& param_blob) {
  dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
  this->LoadParams(&strm);
}

void DecoupledGraphRuntime::LoadParams(dmlc::Stream* strm) {
  if (this->tempParams_ != nullptr) {
    delete this->tempParams_;
    this->tempParams_ = nullptr;
  }

  if (this->copyParamsToEIDs_ != nullptr) {
    delete this->copyParamsToEIDs_;
    this->copyParamsToEIDs_ = nullptr;
  }
  uint64_t header, reserved;
  CHECK(strm->Read(&header))
      << "Invalid parameters file format";
  CHECK(header == kTVMNDArrayListMagic)
      << "Invalid parameters file format";
  CHECK(strm->Read(&reserved))
      << "Invalid parameters file format";

  std::vector<std::string> names;
  CHECK(strm->Read(&names))
      << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  paramsSize_ = static_cast<size_t>(sz);
  CHECK(paramsSize_ == names.size())
      << "Invalid parameters file format";

  this->tempParams_ = new NDArray*[paramsSize_];
  this->copyParamsToEIDs_ = new uint32_t[paramsSize_];

  for (size_t i = 0; i < paramsSize_; ++i) {
    int in_idx = GetInputIndex(names[i]);
    CHECK_GE(in_idx, 0) << "Found param for non-existent input: " << names[i];
    uint32_t eid = this->entry_id(input_nodes_[in_idx], 0);
    //CHECK_LT(eid, data_entry_.size());

    // The data_entry is allocated on device, NDArray.load always load the array into CPU.
    this->tempParams_[i] = new NDArray();
    this->tempParams_[i]->Load(strm);
    this->copyParamsToEIDs_[i] = eid;
  }
}

/*!
 * \brief Upload previously prepped paramters to device.
 * \param param_blob A binary blob of parameter.
 */
void DecoupledGraphRuntime::UploadParams() {
  // CHECK(tempParams_ != nullptr) << "load_params not called before load_to_device";
  auto load_start = std::chrono::high_resolution_clock::now();
  if (tempParams_ != nullptr) {
    for (size_t i = 0; i < this->paramsSize_; ++i) {
      uint32_t eid = copyParamsToEIDs_[i];
      CHECK_LT(eid, data_entry_.size());
      data_entry_[eid].CopyFrom(*this->tempParams_[i]);
    }
  } else {
    CHECK(contiguous_input_memory.size > 0) << "contiguous memory not setup correctly";
    contiguous_input_memory.backing_array_params_view.CopyFrom(*contiguous_input_memory.tempParamsArray);
  }
  auto load_end = std::chrono::high_resolution_clock::now();
  auto  load_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(load_end - load_start);
  // std::cout << "Uploading took " << load_dur.count() << " nanoseconds\n";
}

/*!
 * \brief Do necessary transfers to device.
 * \param param_blob A binary blob of parameter.
 */
void* DecoupledGraphRuntime::LoadToDevice() {
  // Allocate and assign the storage we need on the device as previously
  // calculated in SetupStorage or SetupStorageContiguous
  this->AllocateStorageSpace();

  // Now that arguments of device functions have addresses, we can setup the
  // the graph's computations as an array of TVMOps to execute
  this->SetupOpExecs();

  // With storage allocated, we can upload the parameters previously loaded
  // from a binary blob into the appropriate memory locations on device
  this->UploadParams();

  return (contiguous_input_memory.size > 0) ? contiguous_input_memory.backing_array.dataptr() : nullptr;
}

NDArray DecoupledGraphRuntime::GetConstParams() {
  NDArray gpu_contiguous = this->contiguous_input_memory.backing_array_params_view;
  return gpu_contiguous.CopyTo({kDLCPU, 0});
}

void DecoupledGraphRuntime::SetConstParams(NDArray params) {
  params.CopyTo(this->contiguous_input_memory.backing_array_params_view);
}

void DecoupledGraphRuntime::SetupStorage() {
  std::vector<TVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(tvm::runtime::String2TVMType(s_type));
  }

  // Size and device type of each storage pool entry.
  std::vector<PoolEntry> pool_entries_;
  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    // Use the fallback device if no device index is available.
    int device_type = static_cast<int>(ctxs_[0].device_type);
    if (!attrs_.device_index.empty()) {
      device_type = attrs_.device_index[i];
    }
    size_t size = 1;
    for (int64_t sz : attrs_.shape[i]) {
      size *= static_cast<size_t>(sz);
    }
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op";
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    CHECK(bits % 8U ==  0U || bits ==1U);
    size_t bytes = ((bits + 7U) / 8U) * size;

    uint32_t sid = static_cast<uint32_t>(storage_id);
    if (sid >= pool_entries_.size()) {
      pool_entries_.resize(sid + 1, {0, -1});
    } else {
      CHECK(pool_entries_[sid].device_type == -1 ||
            pool_entries_[sid].device_type == device_type)
          << "The same pool entry cannot be assigned to multiple devices";
    }
    pool_entries_[sid].size = std::max(pool_entries_[sid].size, bytes);
    pool_entries_[sid].device_type = device_type;
  }
}

void DecoupledGraphRuntime::AllocateStorageSpace() {
  // Get all the datatypes for the data entries
  std::vector<TVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(tvm::runtime::String2TVMType(s_type));
  }

  if (contiguous_input_memory.size == 0) {
    // Allocate the space.
    for (const auto& pit : pool_entries_) {
      std::vector<int64_t> shape;
      // This for loop is very fast since there are usually only a couple of
      // devices available on the same hardware.
      const auto& cit =
          std::find_if(ctxs_.begin(), ctxs_.end(), [&pit](const TVMContext& c) {
            return pit.device_type == static_cast<int>(c.device_type);
          });
      TVMContext ctx = cit == ctxs_.end() ? ctxs_[0] : *cit;
      shape.push_back(static_cast<int64_t>(pit.size + 3) / 4);
      storage_pool_.push_back(
          NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx));
    }
  } else {
    // create contiguous memory
    contiguous_memory_allocation &contiguous = this->contiguous_input_memory;
    contiguous.backing_array = NDArray::Empty(contiguous.size, DLDataType{kDLFloat, 32, 1}, contiguous.ctx);


    // Create a view onto the contiguous memory of just the const params
    std::vector<int64_t> paramsshape{{static_cast<int64_t>(contiguous.paramsSize)}};
    contiguous.backing_array_params_view = contiguous.backing_array.CreateView(0, paramsshape, DLDataType{kDLFloat, 32, 1});
    

    // Create views onto the contiguous storage
    for (unsigned i = 0; i < contiguous.storage_ids.size(); i++) {
      std::vector<int64_t> shape;
      // compute shape by difference of offsets in contiguous mem
      shape.push_back(static_cast<int64_t>(contiguous.offsets[i+1] - contiguous.offsets[i]) / 4);
      size_t byte_offset = static_cast<size_t>(contiguous.offsets[i]);
      storage_pool_[contiguous.storage_ids[i]] = contiguous.backing_array.CreateView(byte_offset, shape, DLDataType{kDLFloat, 32, 1});
    }

    // Allocate regular storage
    for (PoolEntry &pe : pool_entries_) {
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>((pe.size + 3) / 4));

      // This for loop is very fast since there are usually only a couple of
      // devices available on the same hardware.
      const auto& cit =
          std::find_if(ctxs_.begin(), ctxs_.end(), [&pe](const TVMContext& c) {
            return pe.device_type == static_cast<int>(c.device_type);
          });
      TVMContext ctx = cit == ctxs_.end() ? ctxs_[0] : *cit;
      storage_pool_[pe.storage_id] = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx);
    }
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  data_entry_.resize(num_node_entries());
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    CHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    data_entry_[i] =
        storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);
  }
}

void DecoupledGraphRuntime::SetupStorageContiguous() {
  typedef struct data_entry_spec{
    unsigned id;
    std::vector<int64_t> shape;
    TVMType dtype;
    size_t size;
    int storage_id;
    int device_type;
    bool is_input;
    TVMContext ctx;
  } data_entry_spec;

  std::vector<data_entry_spec> ds(attrs_.shape.size());

  for (unsigned i = 0; i < ds.size(); i++) {
    data_entry_spec &d = ds[i];
    d.id = i;
    d.shape = this->attrs_.shape[i];
    d.dtype = tvm::runtime::String2TVMType(attrs_.dltype[i]);
    d.size = GetDataSize(d.shape, d.dtype);
    d.storage_id = attrs_.storage_id[i];
    d.device_type = this->attrs_.device_index.empty() ? static_cast<int>(this->ctxs_[0].device_type) : this->attrs_.device_index[i];
    d.is_input = false;


    const auto& cit =
        std::find_if(ctxs_.begin(), ctxs_.end(), [&d](const TVMContext& c) {
          return d.device_type == static_cast<int>(c.device_type);
        });
    d.ctx = cit == ctxs_.end() ? ctxs_[0] : *cit;
  }

  for (uint32_t input_node_id : this->input_nodes_) {
    ds[input_node_id].is_input = true;
  }
  ds[0].is_input = false; // "data" input isn't an input node.  hard code assuming input 0 is data, but should fix later TODO

  typedef struct storage_spec {
    storage_spec() : is_input(false), size(0) {}
    int storage_id;
    std::vector<data_entry_spec> data_entries;
    bool is_input;
    size_t size;
    int device_type;
    TVMContext ctx;
  } storage_spec;

  // TODO: change to a vector, storage_id is always <= storage_pool_.size()
  std::unordered_map<int, storage_spec> specs;

  for (data_entry_spec &d : ds) {
    if (specs.find(d.storage_id) != specs.end()) {
      int device_type = specs[d.storage_id].device_type;
      CHECK(device_type == -1 || device_type == d.device_type) << "The same pool entry cannot be assigned to multiple devices";
    }

    storage_spec &s = specs[d.storage_id];
    s.storage_id = d.storage_id;
    s.data_entries.push_back(d);
    s.is_input |= d.is_input;
    s.size = std::max(s.size, d.size);
    s.device_type = d.device_type;
    s.ctx = d.ctx;
  }


  // Divide into contiguous GPU storage and the rest (TODO: generalize to any ctxs)
  std::vector<storage_spec> gpu_contiguous_inputs;
  std::vector<storage_spec> gpu_contiguous_other;

  for (const auto &e : specs) {
    const storage_spec &s = e.second;
    if (s.device_type == kDLGPU) {
      if (s.is_input) {
        gpu_contiguous_inputs.push_back(s);
      } else {
        gpu_contiguous_other.push_back(s);        
      }
    } else {
      PoolEntry pe(s.size, s.device_type, s.storage_id);
      pool_entries_.push_back(pe);
    }
  }

  // Set up the storage pool
  storage_pool_.resize(specs.size());

  // Describe the GPU contiguous storage
  contiguous_memory_allocation &contiguous = this->contiguous_input_memory;
  for (storage_spec &s : gpu_contiguous_inputs) {
    contiguous.storage_ids.push_back(s.storage_id);
    contiguous.offsets.push_back(contiguous.size * 4);
    contiguous.size += (s.size + 3) / 4;
  }

  // Mark the end of the const params
  contiguous.paramsSize = contiguous.size;

  // Now add inputs
  for (storage_spec &s : gpu_contiguous_other) {
    contiguous.storage_ids.push_back(s.storage_id);
    contiguous.offsets.push_back(contiguous.size * 4);
    contiguous.size += (s.size + 3) / 4;
  }

  // push_back one more offset for size calculations without explicitly saving them
  contiguous.offsets.push_back(contiguous.size * 4);
  contiguous.ctx = gpu_contiguous_inputs[0].ctx;
}

void DecoupledGraphRuntime::SetupOpExecs() {
  op_execs_.resize(this->GetNumOfNodes());
  // setup the array and requirements.
  // std::cout << "Setting up " << this->GetNumOfNodes() << " OpExecs" << std::endl;
  for (uint32_t nid = 0; nid < this->GetNumOfNodes(); ++nid) {
    const auto& inode = nodes_[nid];
    if (inode.op_type == "null") continue;
    std::vector<DLTensor> args;
    // std::cout << inode.name << std::endl;
    // std::cout << "  " << inode.inputs.size() << " inputs" << std::endl;
    for (const auto& e : inode.inputs) {
      args.push_back(*(data_entry_[this->entry_id(e)].operator->()));
    }
    // std::cout << "  " << inode.param.num_outputs << " outputs " << std::endl;
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(*(data_entry_[eid].operator->()));
    }
    CHECK(inode.op_type == "tvm_op") << "Can only take tvm_op as op";

    op_execs_[nid] = CreateTVMOp(inode.param, args, inode.inputs.size());
  }
}

std::function<void()> DecoupledGraphRuntime::CreateTVMOp(
    const TVMOpParam& param,
    const std::vector<DLTensor>& args,
    size_t num_inputs) {
  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<TVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = std::move(args);
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    TVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }

  if (param.func_name == "__nop") {
    return [](){};
  } else if (param.func_name == "__copy") {
    // Perform cross device data copy.
    // Directly copy data from the input to the output.
    auto& name = param.func_name;
    auto fexec = [arg_ptr, name]() {
      // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: LIB NAMED_FUNC_CALL " << name << std::endl;
      DLTensor* from = static_cast<DLTensor*>(arg_ptr->arg_values[0].v_handle);
      DLTensor* to = static_cast<DLTensor*>(arg_ptr->arg_values[1].v_handle);
      TVM_CCALL(TVMArrayCopyFromTo(from, to, nullptr));
    };
    return fexec;
  }

  // Get compiled function from the module that contains both host and device
  // code.
  tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, false);
  CHECK(pf != nullptr) << "no such function in module: " << param.func_name;
  auto& name = param.func_name;
  auto fexec = [arg_ptr, pf, name]() {
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: LIB NAMED_FUNC_CALL " << name << std::endl;
    TVMRetValue rv;
    TVMArgs targs(arg_ptr->arg_values.data(),
                  arg_ptr->arg_tcodes.data(),
                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
    // std::cout << "DYNAMIC_FUNC_EXIT" << std::endl;
  };
  return fexec;
}



clockwork::model::ModelDef* DecoupledGraphRuntime::ExtractModelSpec() {
  // First figure out storage offsets
  typedef struct data_entry_spec{
    unsigned id;
    std::vector<int64_t> shape;
    TVMType dtype;
    size_t size;
    int storage_id;
    int device_type;
    bool is_input;
    TVMContext ctx;
  } data_entry_spec;

  std::vector<data_entry_spec> ds(attrs_.shape.size());

  for (unsigned i = 0; i < ds.size(); i++) {
    data_entry_spec &d = ds[i];
    d.id = i;
    d.shape = this->attrs_.shape[i];
    d.dtype = tvm::runtime::String2TVMType(attrs_.dltype[i]);
    d.size = GetDataSize(d.shape, d.dtype);
    d.storage_id = attrs_.storage_id[i];
    d.device_type = this->attrs_.device_index.empty() ? static_cast<int>(this->ctxs_[0].device_type) : this->attrs_.device_index[i];
    d.is_input = false;


    const auto& cit =
        std::find_if(ctxs_.begin(), ctxs_.end(), [&d](const TVMContext& c) {
          return d.device_type == static_cast<int>(c.device_type);
        });
    d.ctx = cit == ctxs_.end() ? ctxs_[0] : *cit;
  }

  for (uint32_t input_node_id : this->input_nodes_) {
    ds[input_node_id].is_input = true;
  }
  ds[0].is_input = false; // Assume "data" is first input, exclude it from contiguous

  typedef struct storage_spec {
    storage_spec() : is_input(false), size(0) {}
    int storage_id;
    std::vector<data_entry_spec> data_entries;
    bool is_input;
    size_t size;
    int device_type;
    TVMContext ctx;
    size_t offset;
  } storage_spec;

  std::unordered_map<int, storage_spec> specs;
  std::vector<int> spec_order;

  for (data_entry_spec &d : ds) {
    if (specs.find(d.storage_id) != specs.end()) {
      int device_type = specs[d.storage_id].device_type;
      CHECK(device_type == -1 || device_type == d.device_type) << "The same pool entry cannot be assigned to multiple devices";
    } else {
      spec_order.push_back(d.storage_id);
    }

    storage_spec &s = specs[d.storage_id];
    s.storage_id = d.storage_id;
    s.data_entries.push_back(d);
    s.is_input |= d.is_input;
    s.size = std::max(s.size, d.size);
    s.device_type = d.device_type;
    s.ctx = d.ctx;
    s.offset = 0;
  }


  // Divide into contiguous GPU storage and the rest (TODO: generalize to any ctxs)
  std::vector<storage_spec> gpu_contiguous_inputs;
  std::vector<storage_spec> gpu_contiguous_other;
  std::vector<storage_spec> non_gpu;
  for (int &storage_id : spec_order) {
    const storage_spec &s = specs[storage_id];
    if (s.is_input && s.device_type == kDLGPU) gpu_contiguous_inputs.push_back(s);
    else if (s.device_type == kDLGPU) gpu_contiguous_other.push_back(s);
    else non_gpu.push_back(s);
  }


  // Calculate offsets
  uint64_t total_contiguous_size = 0;
  for (storage_spec &s : gpu_contiguous_inputs) {
    specs[s.storage_id].offset = total_contiguous_size;
    total_contiguous_size += 4 * ((s.size + 3) / 4);
  }
  uint64_t weights_size = total_contiguous_size;
  for (storage_spec &s : gpu_contiguous_other) {
    specs[s.storage_id].offset = total_contiguous_size;
    total_contiguous_size += 4 * ((s.size + 3) / 4);
  }

  if (non_gpu.size() > 0) {
    std::cout << "ERROR THERE IS A NON-GPU STORAGE SPEC" << std::endl;
  }


  clockwork::model::ModelDef* mm = new clockwork::model::ModelDef();
  mm->total_memory = 256 * ((total_contiguous_size+255)/256);
  mm->weights_memory = weights_size;


  std::unordered_map<std::string, unsigned> so_functions;
  std::unordered_map<std::string, unsigned> cuda_functions;

  int skipped = 0;
  uint64_t max_workspace_memory = 0;
  for (uint32_t nid = 0; nid < this->GetNumOfNodes(); ++nid) {
    const auto& inode = nodes_[nid];
    if (inode.op_type == "null") {
      skipped++;
      continue;
    };

    clockwork::model::OpDef op;

    // Get the DLTensor sizes for this op
    std::vector<DLTensor> args;

    for (const auto& e : inode.inputs) {
      storage_spec &spec = specs[ds[this->entry_id(e)].storage_id];
      clockwork::model::DLTensorDef d;
      d.offset = spec.offset;
      d.size = spec.size;
      d.shape = this->attrs_.shape[this->entry_id(e)];
      op.inputs.push_back(d);
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      storage_spec &spec = specs[ds[eid].storage_id];
      clockwork::model::DLTensorDef d;
      d.offset = spec.offset;
      d.size = spec.size;
      d.shape = this->attrs_.shape[eid];
      op.inputs.push_back(d);
    }

    // Get the name of the function from the SO that this op invokes
    auto search = so_functions.find(inode.param.func_name);
    if (search == so_functions.end()) {
      so_functions[inode.param.func_name] = mm->so_functions.size();
      mm->so_functions.push_back(inode.param.func_name);
    }
    op.so_function = so_functions[inode.param.func_name];

    // Run the operation to get workspace alloc size
    ManagedCUDADeviceAPI::Global()->tracker.enabled = true;
    op_execs_[nid]();
    std::vector<WorkspaceAlloc> allocs = ManagedCUDADeviceAPI::Global()->tracker.get();
    std::cout << "Op " << nid << " had " << allocs.size() << "workspace allocs (" << inode.param.func_name << ")" << std::endl;

    int currentAllocOffset = 0;
    for (unsigned i = 0; i < allocs.size(); i++) {
      if (allocs[i].isalloc) {
        clockwork::model::WorkspaceAllocDef d;
        d.offset = mm->total_memory + currentAllocOffset;
        d.size = 256 * ((allocs[i].size+255)/256);
        op.workspace_allocs.push_back(d);
        currentAllocOffset += d.size; // Align to 256?
      }
    }
    if (currentAllocOffset > max_workspace_memory) {
      max_workspace_memory = currentAllocOffset;
    }

    ManagedCUDADeviceAPI::Global()->tracker.clear();
    ManagedCUDADeviceAPI::Global()->tracker.enabled = false;

    mm->ops.push_back(op);
  }
  mm->workspace_memory = max_workspace_memory;
  mm->total_memory += max_workspace_memory;

  // Get inputs and outputs data

  for (unsigned i = 0; i < outputs_.size(); i++) {
    uint32_t eid = this->entry_id(outputs_[i]);
    storage_spec &spec = specs[ds[eid].storage_id];
    clockwork::model::DLTensorDef d;
    d.offset = spec.offset;
    d.size = spec.size;
    d.shape = this->attrs_.shape[eid];
    mm->outputs.push_back(d);
  }
  
  // Assume the only input is "data" at ds[0]
  {
    storage_spec &spec = specs[ds[0].storage_id];
    clockwork::model::DLTensorDef d;
    d.offset = spec.offset;
    d.size = spec.size;
    d.shape = this->attrs_.shape[0];
    mm->inputs.push_back(d);
  }

  std::cout << "model has " << mm->total_memory << " total memory" << std::endl;
  std::cout << "model has " << mm->weights_memory << " weights memory" << std::endl;
  std::cout << "model has " << mm->workspace_memory << " intra-op memory" << std::endl;
  std::cout << "model has " << (total_contiguous_size - mm->weights_memory) << " inter-op memory" << std::endl;
  std::cout << "model has " << mm->so_functions.size() << " so functions" << std::endl;
  std::cout << "model has " << mm->ops.size() << " ops" << std::endl;
  std::cout << "model has " << mm->inputs.size() << " inputs" << std::endl;
  for (unsigned i = 0; i < mm->inputs.size(); i++) {
    std::cout << "  input " << i << " offset " << mm->inputs[i].offset << " size " << mm->inputs[i].size << " shape [ ";
    for (unsigned j = 0; j < mm->inputs[i].shape.size(); j++) {
      std::cout << mm->inputs[i].shape[j] << " ";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "model has " << mm->outputs.size() << " outputs" << std::endl;
  for (unsigned i = 0; i < mm->outputs.size(); i++) {
    std::cout << "  output " << i << " offset " << mm->outputs[i].offset << " size " << mm->outputs[i].size << " shape [ ";
    for (unsigned j = 0; j < mm->outputs[i].shape.size(); j++) {
      std::cout << mm->outputs[i].shape[j] << " ";
    }
    std::cout << "]" << std::endl;
  }

  return mm;
}

std::string DecoupledGraphRuntime::SaveParams() {
  std::string* outString = new std::string();
  dmlc::MemoryStringStream strm(const_cast<std::string*>(outString));
  auto params = this->GetConstParams();
  params.Save(&strm);
  return *outString;
}

void DecoupledGraphRuntime::LoadParamsContiguously(const std::string& param_blob) {
  if (this->contiguous_input_memory.tempParamsArray != nullptr) {
    delete this->contiguous_input_memory.tempParamsArray;
    this->contiguous_input_memory.tempParamsArray = nullptr;
  }
  dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
  this->contiguous_input_memory.tempParamsArray = new NDArray();
  this->contiguous_input_memory.tempParamsArray->Load(&strm);
}

PackedFunc DecoupledGraphRuntime::GetFunction(
  const std::string& name,
  const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          if (in_idx >= 0) this->SetInput(in_idx, args[1]);
        } else {
          this->SetInput(args[0], args[1]);
        }
      // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        if (args.num_args == 2) {
          this->CopyOutputTo(args[0], args[1]);
        } else {
          *rv = this->GetOutput(args[0]);
        }
      // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        int in_idx = 0;
        if (args[0].type_code() == kStr) {
          in_idx = this->GetInputIndex(args[0]);
        } else {
          in_idx = args[0];
        }
        CHECK_GE(in_idx, 0);
        *rv = this->GetInput(in_idx);
      // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "get_num_outputs") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        *rv = this->NumOutputs();
      // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        this->Run();
      // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "load_params") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        const std::string s = args[0].operator std::string();
        this->LoadParams(s);
        // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "load_params_contig") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        //const std::string s = args[0].operator std::string();
        //// std::cout << "length in packed func: " << s.length() << std::endl;
        this->LoadParamsContiguously(args[0]);
        // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "get_const_params") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        *rv = this->GetConstParams();
        // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "set_const_params") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        this->SetConstParams(args[0]);
        // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "extract_model_spec") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->ExtractModelSpec();
      });
  } else if (name == "extract_workspace_allocs") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->ExtractWorkspaceAllocs();
      });
  } else if (name == "load_to_device") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        *rv = this->LoadToDevice();
        // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "evicted") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        this->contiguous_input_memory.backing_array.evict();
        // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "get_contig_context") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        *rv = this->contiguous_input_memory.ctx;
        // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else if (name == "save_params") {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
      // std::cout << "FUNC_ENTER " << name << std::endl;
        *rv = this->SaveParams();
        // std::cout << "FUNC_EXIT " << name << std::endl;
      });
  } else {
    return PackedFunc();
  }
}

Module DecoupledGraphRuntimeCreate(const std::string& sym_json,
                          const tvm::runtime::Module& m,
                          const std::vector<TVMContext>& ctxs,
                          const bool contiguous) {
  std::shared_ptr<DecoupledGraphRuntime> exec = std::make_shared<DecoupledGraphRuntime>();
  exec->Init(sym_json, m, ctxs, contiguous);
  return Module(exec);
}

Module DecoupledGraphRuntimeCreate(const std::string& sym_json,
                          const tvm::runtime::Module& m, int device_type, int device_id) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  std::vector<TVMContext> ctxs;
  ctxs.push_back(ctx);
  return DecoupledGraphRuntimeCreate(sym_json, m, ctxs, true);
}

std::shared_ptr<DecoupledGraphRuntime> DecoupledGraphRuntimeCreateDirect(const std::string& sym_json,
                          const tvm::runtime::Module& m, int device_type, int device_id) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  std::vector<TVMContext> ctxs;
  ctxs.push_back(ctx);
  std::shared_ptr<DecoupledGraphRuntime> exec = std::make_shared<DecoupledGraphRuntime>();
  exec->Init(sym_json, m, ctxs, true);
  return exec;
}

// Get all context for the host and other runtime devices.
std::vector<TVMContext> GetAllContext(const TVMArgs& args) {
  // Reserve the first item as the fallback device.
  std::vector<TVMContext> ret;
  TVMContext ctx;
  for (int i = 2; i < args.num_args; i += 2) {
    int dev_type = args[i];
    ctx.device_type = static_cast<DLDeviceType>(dev_type);
    ctx.device_id = args[i + 1];
    ret.push_back(ctx);
  }
  return ret;
}

// 4-argument version is currently reserved to keep support of calling
// from tvm4j and javascript, since they don't have heterogeneous
// execution support yet. For heterogenenous execution, at least 5 arguments will
// be passed in. The third one is the number of devices.
// Eventually, we will only probably pass TVMContext for all the languages.
TVM_REGISTER_GLOBAL("tvm.decoupled_graph_runtime.create")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    // std::cout << "FUNC_ENTER CREATE RUNTIME" << std::endl;
    CHECK_GE(args.num_args, 4)
        << "The expected number of arguments for graph_runtime.create is "
           "at least 4, but it has "
        << args.num_args;
    const auto& contexts = GetAllContext(args);
    *rv = DecoupledGraphRuntimeCreate(args[0], args[1], contexts, false);
    // std::cout << "FUNC_EXIT CREATE RUNTIME" << std::endl;
  });

TVM_REGISTER_GLOBAL("tvm.decoupled_graph_runtime.create_contiguous")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    // std::cout << "FUNC_ENTER CREATE CONTIGUOUS RUNTIME" << std::endl;
    CHECK_GE(args.num_args, 4)
        << "The expected number of arguments for graph_runtime.create is "
           "at least 4, but it has "
        << args.num_args;
    const auto& contexts = GetAllContext(args);
    *rv = DecoupledGraphRuntimeCreate(args[0], args[1], contexts, true);
    // std::cout << "FUNC_EXIT CREATE CONTIGUOUS RUNTIME" << std::endl;
  });

TVM_REGISTER_GLOBAL("tvm.decoupled_graph_runtime.remote_create")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    CHECK_GE(args.num_args, 4) << "The expected number of arguments for "
                                  "graph_runtime.remote_create is "
                                  "at least 4, but it has "
                               << args.num_args;
    void* mhandle = args[1];
    const auto& contexts = GetAllContext(args);
    *rv = DecoupledGraphRuntimeCreate(
        args[0], *static_cast<tvm::runtime::Module*>(mhandle), contexts, false);
  });
}  // namespace runtime
}  // namespace tvm
