
#ifndef _CLOCKWORK_MODELDEF_H_
#define _CLOCKWORK_MODELDEF_H_

#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include "dmlc/logging.h"

namespace clockwork {
namespace model {

struct DLTensorDef {
	uint64_t offset;
	std::vector<int64_t> shape;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(offset),
        PODS_MDR(shape)
    )
};

struct OpDef {
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

struct ModelDef {
	uint64_t total_memory;
	uint64_t weights_memory;
    uint64_t workspace_memory;
	std::vector<std::string> so_functions;
	std::vector<std::string> cuda_functions;
	std::vector<OpDef> ops;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(total_memory),
        PODS_MDR(weights_memory),
        PODS_MDR(so_functions),
        PODS_MDR(cuda_functions),
        PODS_MDR(ops)
    )

    static void ReadFrom(const std::string &data, ModelDef &def) {
        pods::InputBuffer in(data.data(), data.size());
        pods::BinaryDeserializer<decltype(in)> deserializer(in);
        pods::Error status = deserializer.load(def);
        CHECK(status == pods::Error::NoError) << "Cannot deserialize minmodel";
    }

    // TODO: currently, src/convert.cpp is the only usage of writing model defs; eventually migrate code here
};

}
}

#endif