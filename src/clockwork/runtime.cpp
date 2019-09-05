#include "clockwork/runtime.h"

namespace clockwork {

std::array<TaskType, 7> TaskTypes = {
	Disk, CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_output, Sync 
};

}