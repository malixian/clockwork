
#ifndef _CLOCKWORK_MODELDEF_H_
#define _CLOCKWORK_MODELDEF_H_

#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include "dmlc/logging.h"

namespace clockwork {
namespace model {

struct WorkspaceAllocDef {
    uint64_t page_number;
    uint64_t offset_in_page;
    uint64_t size;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(page_number),
        PODS_MDR(offset_in_page),
        PODS_MDR(size)
    )
};

struct DLTensorDef {
    unsigned page_number;
    uint64_t offset_in_page;
    uint64_t size;
	std::vector<int64_t> shape;

    PODS_SERIALIZABLE(1,
        PODS_MDR(page_number),
        PODS_MDR(offset_in_page),
        PODS_MDR(size),
        PODS_MDR(shape)
    )
};

struct OpDef {
	std::vector<DLTensorDef> inputs;
	unsigned so_function;
	std::vector<unsigned> cuda_functions;
	std::vector<WorkspaceAllocDef> workspace_allocs;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(inputs),
        PODS_MDR(so_function),
        PODS_MDR(cuda_functions),
        PODS_MDR(workspace_allocs)
    )
};

struct WeightsPageDef {
    uint64_t base_offset;
    uint64_t size;

    PODS_SERIALIZABLE(1,
        PODS_MDR(base_offset),
        PODS_MDR(size)
    )
};

struct ModelDef {
	uint64_t weights_size;
    uint64_t non_weights_size;
    uint64_t workspace_size;

    uint64_t configured_page_size;
    uint64_t num_pages;
    std::vector<WeightsPageDef> weights_pages;

	std::vector<std::string> so_functions;
	std::vector<std::string> cuda_functions;
	std::vector<OpDef> ops;
    std::vector<DLTensorDef> inputs;
    std::vector<DLTensorDef> outputs;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(weights_size),
        PODS_MDR(non_weights_size),
        PODS_MDR(workspace_size),
        PODS_MDR(configured_page_size),
        PODS_MDR(num_pages),
        PODS_MDR(weights_pages),
        PODS_MDR(so_functions),
        PODS_MDR(cuda_functions),
        PODS_MDR(ops),
        PODS_MDR(inputs),
        PODS_MDR(outputs)
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