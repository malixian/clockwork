
#include "clockwork/model.h"
#include "clockwork/modeldef.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <dmlc/logging.h>
#include <cstring>

namespace clockwork{
namespace model {

struct Item {
	uint64_t id;
	uint64_t size;
};

class Bucket {
public:
	uint64_t id;
	std::vector<Item> items;

	static int size(const Bucket &b) {
		int size = 0;
		for (unsigned i = 0; i < b.items.size(); i++) {
			size += b.items[i].size;
		}
		return size;
	}
};



struct greater_than_bucketsize {
	inline bool operator() (const Bucket &b1, const Bucket &b2) {
		return Bucket::size(b1) > Bucket::size(b2);
	}
};

struct greater_than_itemsize {
	inline bool operator() (const Item &i1, const Item &i2) {
		return (i1.size > i2.size);
	}
};

std::vector<Bucket> pack(std::vector<Item> items, int bucketsize) {
	// Sort items in descending order of size
	std::sort(items.begin(), items.end(), greater_than_itemsize());

	// Pack each item into bucket that minimizes remaining space
	std::vector<Bucket> buckets;
	for (unsigned i = 0; i < items.size(); i++) {
		unsigned bucket_id = 0;
		for (; bucket_id < buckets.size(); bucket_id++) {
			if (Bucket::size(buckets[bucket_id]) + items[i].size <= bucketsize) {
				break;
			}
		}
		if (bucket_id == buckets.size()) {
			Bucket b;
			b.id = buckets.size();
			buckets.push_back(b);
		}
		buckets[bucket_id].items.push_back(items[i]);
		std::sort(buckets.begin(), buckets.end(), greater_than_bucketsize());
	}

	int total_item_size = 0;
	for (unsigned i = 0; i < items.size(); i++) {
		total_item_size += items[i].size;
	}

	return buckets;
}

struct less_than_offsets {
	inline bool operator() (const DLTensorDef &d1, const DLTensorDef &d2) {
		return d1.offset < d2.offset;
	}
};

class ModelState {
public:
	std::unordered_map<uint64_t, DLTensorDef*> all_tensordefs;
	int valid_duplicates = 0;
	std::unordered_map<uint64_t, int> tensordef_counts;
	std::vector<DLTensorDef> ordered_defs;

	void addAndCheckDLTensorDef(DLTensorDef &input) {
		DLTensorDef* other = all_tensordefs[input.offset];
		if (other == nullptr) {
			all_tensordefs[input.offset] = &input;
			tensordef_counts[input.offset] = 1;
			ordered_defs.push_back(input);
		} else {
			CHECK(other->size == input.size) << "Found TensorDef with same ptr but different sizes";
			valid_duplicates++;
			tensordef_counts[input.offset]++;
		}
		std::sort(ordered_defs.begin(), ordered_defs.end(), less_than_offsets());
	}

	void checkPageSize(int pagesize) {
		for (unsigned i = 1; i < ordered_defs.size(); i++) {
			CHECK(ordered_defs[i].size <= pagesize) << "Tensordef of size " << ordered_defs[i].size << " is greater than page size " << pagesize;
		}
	}

	void checkOverlappingOffsets() {
		for (unsigned i = 1; i < ordered_defs.size(); i++) {
			CHECK(ordered_defs[i-1].offset + ordered_defs[i-1].size <= ordered_defs[i].offset) << "Found a TensorDef that overlaps with a prevoius one";
		}
	}

	void printState() {
		std::cout << "Checked " << all_tensordefs.size() << " tensordefs"
		          << " with " << valid_duplicates << " valid duplicates "
		          << std::endl;
	}
};

struct Remap {
	uint64_t original_offset;
	uint64_t new_offset;
	unsigned page;
	uint64_t offset_in_page;
	uint64_t size;
};

PageMappedModelDef reorder(ModelDef model, int pagesize, const char* weights, char** newweights) {
	std::cout << "model weights memory is " << model.weights_memory << std::endl;

	PageMappedModelDef newmodel;
	newmodel.minimum_required_memory = model.total_memory;
	newmodel.weights_memory = model.weights_memory;
	newmodel.so_functions = model.so_functions;
	newmodel.cuda_functions = model.cuda_functions;


	// All contiguous inputs must remain contiguous
	ModelState state;
	for (OpDef &op : model.ops) {
		for (DLTensorDef &input : op.inputs) {
			if (input.offset < model.weights_memory) {
				state.addAndCheckDLTensorDef(input);
			}
		}
	}

	int total_weights_size=0;
	std::vector<Item> items;
	for (unsigned i = 0; i < state.ordered_defs.size(); i++) {
		items.push_back(Item{i, state.ordered_defs[i].size});
		total_weights_size += state.ordered_defs[i].size;
	}
	std::vector<Bucket> buckets = pack(items, pagesize);
	CHECK(total_weights_size == model.weights_memory) << "Expected " << model.weights_memory << " weights size, got " << total_weights_size;
	std::cout << "Packed " << total_weights_size << " into " << buckets.size() << " pages of size " << pagesize << " pagetotal=" << (pagesize * buckets.size()) << std::endl;



	std::unordered_map<uint64_t, Remap> remapping;
	int cumulativeOffset = 0;
	for (unsigned i = 0; i < buckets.size(); i++) {
		Bucket bucket = buckets[i];
		PageDef page;
		page.base_offset = cumulativeOffset;

		int page_cumulative_offset = 0;
		for (unsigned j = 0; j < bucket.items.size(); j++) {
			Item item = bucket.items[j];

			Remap r;
			r.original_offset = state.ordered_defs[item.id].offset;
			r.new_offset = cumulativeOffset;
			r.page = i;
			r.offset_in_page = page_cumulative_offset;
			r.size = item.size;
			remapping[r.original_offset] = r;

			cumulativeOffset += item.size;
			page_cumulative_offset += item.size;
		}

		page.size = cumulativeOffset - page.base_offset;
		newmodel.weights_pages.push_back(page);
	}

	*newweights = static_cast<char*>(malloc(model.weights_memory));
	for (auto &remap : remapping) {
		std::memcpy(*newweights+remap.second.new_offset, weights+remap.second.original_offset, remap.second.size);
	}
	std::cout << "Did " << remapping.size() << " remappings" << std::endl;



	ModelState nonweights;
	for (OpDef &op : model.ops) {
		for (DLTensorDef &input : op.inputs) {
			if (input.offset >= model.weights_memory) {
				nonweights.addAndCheckDLTensorDef(input);
			}
		}
		for (DLTensorDef &input : model.inputs) {
			nonweights.addAndCheckDLTensorDef(input);
		}
		for (DLTensorDef &output : model.outputs) {
			nonweights.addAndCheckDLTensorDef(output);
		}
	}

	// Add the non-weights to the remapping, but don't construct page objects
	std::vector<Item> nonweights_items;
	int non_weights_size = 0;
	for (unsigned i = 0; i < nonweights.ordered_defs.size(); i++) {
		nonweights_items.push_back(Item{i, nonweights.ordered_defs[i].size});
		non_weights_size += nonweights.ordered_defs[i].size;
	}
	std::vector<Bucket> nonweightsbuckets = pack(nonweights_items, pagesize);
	std::cout << "Packed " << non_weights_size << " non-weights into " << nonweightsbuckets.size() << " pages of size " << pagesize << " pagetotal=" << (pagesize * nonweightsbuckets.size()) << std::endl;

	int non_weights_page_count = nonweightsbuckets.size();
	for (unsigned i = 0; i < nonweightsbuckets.size(); i++) {
		Bucket bucket = nonweightsbuckets[i];

		int page_cumulative_offset = 0;
		for (unsigned j = 0; j < bucket.items.size(); j++) {
			Item item = bucket.items[j];

			Remap r;
			r.original_offset = nonweights.ordered_defs[item.id].offset;
			r.new_offset = cumulativeOffset;
			r.page = newmodel.weights_pages.size() + i;
			r.offset_in_page = page_cumulative_offset;
			r.size = item.size;
			remapping[r.original_offset] = r;

			page_cumulative_offset += item.size;
			cumulativeOffset += item.size;
		}
	}

	std::cout << newmodel.weights_pages.size() << " weights pages, " << non_weights_page_count << " non-weights pages, and " << remapping.size() << " remappings" << std::endl;

	// Now put together all of the ops including workspace allocs
	int workspace_page_count = 0;
	for (unsigned i = 0; i < model.ops.size(); i++) {
		OpDef op = model.ops[i];
		
		PageMappedOpDef newop;
		newop.so_function = op.so_function;
		newop.cuda_functions = op.cuda_functions;

		for (unsigned j = 0; j < op.inputs.size(); j++) {
			DLTensorDef input = op.inputs[j];
			PageMappedDLTensorDef newinput;
			Remap r = remapping[input.offset];

			newinput.base_offset = r.new_offset;
			newinput.page = r.page;
			newinput.page_offset = r.offset_in_page;
			newinput.size = input.size;
			newinput.shape = input.shape;

			newop.inputs.push_back(newinput);
		}

		// Calculate required number of pages for workspace allocs
		std::vector<Item> workspaceAllocItems;
		for (unsigned j = 0; j < op.workspace_allocs.size(); j++) {
			workspaceAllocItems.push_back(Item{j, op.workspace_allocs[j].size});
		}
		std::vector<Bucket> workspaceAllocBuckets = pack(workspaceAllocItems, pagesize);
		if (workspaceAllocBuckets.size() > workspace_page_count) {
			workspace_page_count = workspaceAllocBuckets.size();
		}

		newop.workspace_allocs.resize(op.workspace_allocs.size());
		for (unsigned j = 0; j < workspaceAllocBuckets.size(); j++) {
			Bucket bucket = workspaceAllocBuckets[j];
			int offset_in_page = 0;
			for (unsigned k = 0; k < bucket.items.size(); k++) {
				Item item = bucket.items[k];

				PageMappedWorkspaceAllocDef newalloc;
				newalloc.page = newmodel.weights_pages.size()+non_weights_page_count+j;
				newalloc.page_offset = offset_in_page;
				newalloc.size = item.size;

				newop.workspace_allocs[item.id] = newalloc;

				offset_in_page += item.size;
			}
		}

		newmodel.ops.push_back(newop);
	}

	// Set the final page count
	newmodel.total_pages = newmodel.weights_pages.size() + non_weights_page_count + workspace_page_count;
	newmodel.configured_page_size = pagesize;
	newmodel.paged_required_memory = newmodel.total_pages * newmodel.configured_page_size;

	// Finish with inputs and outputs
	for (unsigned i = 0; i < model.inputs.size(); i++) {
		DLTensorDef input = model.inputs[i];
		PageMappedDLTensorDef newinput;
		Remap r = remapping[input.offset];

		newinput.base_offset = r.new_offset;
		newinput.page = r.page;
		newinput.page_offset = r.offset_in_page;
		newinput.size = input.size;
		newinput.shape = input.shape;

		newmodel.inputs.push_back(newinput);
	}
	for (unsigned i = 0; i < model.outputs.size(); i++) {
		DLTensorDef output = model.outputs[i];
		PageMappedDLTensorDef newoutput;
		Remap r = remapping[output.offset];

		newoutput.base_offset = r.new_offset;
		newoutput.page = r.page;
		newoutput.page_offset = r.offset_in_page;
		newoutput.size = output.size;
		newoutput.shape = output.shape;

		newmodel.outputs.push_back(newoutput);
	}

	return newmodel;
}

void check(ModelDef model, int pagesize) {
	ModelState state;
	for (OpDef &op : model.ops) {
		for (DLTensorDef &input : op.inputs) {
			state.addAndCheckDLTensorDef(input);
		}
	}
	for (DLTensorDef &input : model.inputs) {
		state.addAndCheckDLTensorDef(input);
	}
	for (DLTensorDef &output : model.outputs) {
		state.addAndCheckDLTensorDef(output);
	}
	state.checkOverlappingOffsets();
	state.printState();
	state.checkPageSize(pagesize);
}

void printTensorDef(PageMappedDLTensorDef def, std::string prefix) {
	std::cout << prefix << def.base_offset << " = [" << def.page << " " << def.page_offset << "] + " << def.size << " shape=[ ";
	for (unsigned i = 0; i < def.shape.size(); i++) {
		std::cout << def.shape[i] << " ";
	}
	std::cout << " ]" << std::endl;
}

void printWorkspaceAlloc(PageMappedWorkspaceAllocDef def, std::string prefix) {
	std::cout << prefix << "[" << def.page << " " << def.page_offset << "] + " << def.size << std::endl;
}

void printOp(PageMappedOpDef op, std::string prefix) {
	std::cout << prefix << op.so_function << ":" << std::endl;
	for (unsigned i = 0; i < op.inputs.size(); i++) {
		printTensorDef(op.inputs[i], prefix+"   ");
	}
	if (op.workspace_allocs.size() > 0) {
		std::cout << prefix << "  " << "Workspace:" << std::endl;
		for (unsigned i = 0; i < op.workspace_allocs.size(); i++) {
			printWorkspaceAlloc(op.workspace_allocs[i], prefix+"    ");
		}
	}
}

void printPageDef(PageDef def, std::string prefix) {
	std::cout << prefix << "[" << def.base_offset << " +" << def.size << "]" << std::endl;
}

void printNewModel(PageMappedModelDef model) {
	std::cout << std::endl << "------------------ NEW MODEL ------------------" << std::endl;
	std::cout << model.paged_required_memory << " required memory in paged-mode" << std::endl;
	std::cout << model.minimum_required_memory << " required memory in non-paged mode (min necessary)" << std::endl;
	std::cout << model.weights_memory << " total weights memory" << std::endl;
	std::cout << (model.configured_page_size * model.weights_pages.size()) << " total weights paged on " << model.weights_pages.size() << " pages" << std::endl;
	std::cout << model.total_pages << " pages of size " << model.configured_page_size << " needed" << std::endl;
	std::cout << model.so_functions.size() << " SO functions and " << model.cuda_functions.size() << " CUDA functions" << std::endl;
	std::cout << model.ops.size() << " ops:" << std::endl;
	for (unsigned i = 0; i < model.ops.size(); i++) {
		printOp(model.ops[i], "   ");
	}
	std::cout << "Inputs:" << std::endl;
	for (unsigned i = 0; i < model.inputs.size(); i++) {
		printTensorDef(model.inputs[i], "   ");
	}
	std::cout << "Outputs:" << std::endl;
	for (unsigned i = 0; i < model.outputs.size(); i++) {
		printTensorDef(model.outputs[i], "   ");
	}
	std::cout << "Weights pages:" << std::endl;
	for (unsigned i = 0; i < model.weights_pages.size(); i++) {
		printPageDef(model.weights_pages[i], "   ");
	}

}

PageMappedModelDef processModelDef(ModelDef def, int pagesize, const char* weights, char** newweights) {
	check(def, pagesize);
	PageMappedModelDef newmodel = reorder(def, pagesize, weights, newweights);
	printNewModel(newmodel);
	return newmodel;
}

}
}