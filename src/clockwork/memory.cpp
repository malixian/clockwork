#include "clockwork/memory.h"
#include <dmlc/logging.h>

namespace clockwork {

PageCache::PageCache(char* baseptr, size_t total_size, size_t page_size) : size(total_size), baseptr(baseptr), page_size(page_size), n_pages(total_size/page_size) {
	CHECK(total_size % page_size == 0) << "Cannot create page cache -- page_size " << page_size << " does not equally divide total_size " << total_size;

	// Construct and link pages
	for (unsigned i = 0; i < n_pages; i++) {
		Page* p = new Page();
		p->ptr = baseptr + i * page_size;
		p->current_allocation = nullptr;
		freePages.pushBack(p);
	}
}

/* 
Locks the allocation if it hasn't been evicted
*/
bool PageCache::trylock(std::shared_ptr<Allocation> allocation) {
	std::lock_guard<std::mutex> lock(mutex);

	if (allocation->evicted) {
		return false;
	}

	if (allocation->usage_count++ == 0) {
		// Lock the allocation
		unlockedAllocations.remove(allocation->list_position);
		allocation->list_position = lockedAllocations.pushBack(allocation); // Tracking locked allocations is probably unnecessary
	}

	return true;
}

/* 
Locks the allocation; error if it's evicted
*/
void PageCache::lock(std::shared_ptr<Allocation> allocation) {
	CHECK(trylock(allocation)) << "Cannot lock evicted allocation";
}

void PageCache::unlock(std::shared_ptr<Allocation> allocation) {
	std::lock_guard<std::mutex> lock(mutex);

	CHECK(!(allocation->evicted)) << "Tried unlocking an allocation that's already been evicted";

	if (--allocation->usage_count == 0) {
		// Unlock the allocation
		lockedAllocations.remove(allocation->list_position);
		allocation->list_position = unlockedAllocations.pushBack(allocation);
	}
}

std::shared_ptr<Allocation> PageCache::alloc(unsigned n_pages, EvictionCallback* callback) {
	std::shared_ptr<Allocation> alloc = std::make_shared<Allocation>();
	alloc->callback = callback;
	alloc->pages.reserve(n_pages);

	std::lock_guard<std::mutex> lock(mutex);

	// Use up free pages
	while(alloc->pages.size() < n_pages && !freePages.isEmpty()) {
		Page* p = freePages.popHead();
		p->current_allocation = alloc;
		alloc->pages.push_back(p);
	}

	// Start evicting allocations
	std::vector<EvictionCallback*> callbacks;
	while (alloc->pages.size() < n_pages && !unlockedAllocations.isEmpty()) {
		std::shared_ptr<Allocation> toEvict = unlockedAllocations.popHead();
		toEvict->evicted = true;
		callbacks.push_back(toEvict->callback);

		unsigned i = 0;

		// Claim as many of the evicted pages as we need
		for (; i < toEvict->pages.size() && alloc->pages.size() < n_pages; i++) {
			Page* p = toEvict->pages[i];
			p->current_allocation = alloc;
			alloc->pages.push_back(p);
		}

		// Put the remaining evicted pages in the list of free pages
		for (; i < toEvict->pages.size(); i++) {
			Page* p = toEvict->pages[i];
			p->current_allocation = nullptr;
			freePages.pushBack(p);		
		}
	}

	// Free alloced pages if this alloc is going to fail
	if (alloc->pages.size() < n_pages) {
		// If we reach here, we were unable to alloc enough pages,
		// because too many allocations are locked and cannot be evicted
		// This case could be optimized but for now don't
		// Put back all of the free pages we took
		for (unsigned i = 0; i < alloc->pages.size(); i++) {
			Page* p = alloc->pages[i];
			p->current_allocation = nullptr;
			freePages.pushBack(p);
		}
		alloc = nullptr;
	} else {
		// Allocation successful; lock it
		alloc->usage_count++;
		alloc->list_position = lockedAllocations.pushBack(alloc);
	}

	// Notify eviction handlers
	for (unsigned i = 0; i < callbacks.size(); i++) {
		if (callbacks[i] != nullptr) {
			callbacks[i]->evicted();
		}
	}

	return alloc;
}

void PageCache::free(std::shared_ptr<Allocation> allocation) {
	std::lock_guard<std::mutex> lock(mutex);

	CHECK(!(allocation->evicted)) << "Tried freeing an allocation that's already been evicted";
	CHECK(allocation->usage_count == 0) << "Tried freeing an allocation that's currently in use";

	// Remove from the unlocked allocations
	unlockedAllocations.remove(allocation->list_position);

	// Free all the pages
	for (unsigned i = 0; i < allocation->pages.size(); i++) {
		Page* p = allocation->pages[i];
		p->current_allocation = nullptr;
		freePages.pushBack(p);
	}

	// Mark as evicted but don't call eviction callback since it was explicitly freed
	allocation->evicted = true;
}

}