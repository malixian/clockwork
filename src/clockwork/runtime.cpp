#include "clockwork/runtime.h"

namespace clockwork {

std::array<TaskType, 7> TaskTypes = {
	Disk, CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_Output, Sync 
};

std::string TaskTypeName(TaskType type) {
	switch(type) {
		case Disk: return "Disk";
		case CPU: return "CPU";
		case PCIe_H2D_Weights: return "PCIe_H2D_Weights";
		case PCIe_H2D_Inputs: return "PCIe_H2D_Weights";
		case GPU: return "GPU";
		case PCIe_D2H_Output: return "PCIe_D2H_Output";
		case Sync: return "Sync";
	};
}

}
