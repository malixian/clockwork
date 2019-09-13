#ifndef _CLOCKWORK_MODEL_MEMORY_H_
#define _CLOCKWORK_MODEL_MEMORY_H_

#include <mutex>
#include "clockwork/util/util.h"
#include <memory>
#include <atomic>

namespace clockwork {
namespace model {

template<typename T> class LinkedList;

template<typename T> class LinkedListElement {
public:
	LinkedListElement<T>* next = nullptr;
	LinkedListElement<T>* prev = nullptr;
	LinkedList<T>* container = nullptr;
	T data;
	LinkedListElement(T &data) : data(data) {}
};

template<typename T> class LinkedList {
private:
	LinkedListElement<T>* head = nullptr;
	LinkedListElement<T>* tail = nullptr;
public:

	bool isEmpty() {
		return head==nullptr;
	}

	T popHead() {
		LinkedListElement<T>* elem = head;
		if (elem != nullptr) head = head->next;
		if (head == nullptr && tail == elem) tail = nullptr;
		T data = elem->data;
		delete elem;
		return data;
	}

	T popTail() {
		LinkedListElement<T>* elem = tail;
		if (elem != nullptr) tail = tail->prev;
		if (tail == nullptr && head == elem) head = nullptr;
		T data = elem->data;
		delete elem;
		return data;
	}

	void remove(LinkedListElement<T>* element) {
		CHECK(element->container == this) << "Cannot remove linked list element from list it's not part of";
		if (element->next != nullptr) element->next->prev = element->prev;
		else if (tail == element) tail = element->prev;
		if (element->prev != nullptr) element->prev->next = element->next;
		else if (head == element) head = element->next;
		delete element;
	}

	LinkedListElement<T>* pushBack(T data) {
		LinkedListElement<T>* element = new LinkedListElement<T>(data);
		if (head == nullptr) {
			head = element;
			tail = element;
			element->next = nullptr;
			element->prev = nullptr;
			element->container = this;
		} else {
			element->prev = tail;
			tail->next = element;
			tail = element;
			element->container = this;
		}
		return element;
	}	
};

class EvictionCallback {
public:
	virtual void evicted() = 0;
};

struct Page;

struct Allocation {
	bool evicted = false;
	int usage_count = 0;
	std::vector<Page*> pages;
	EvictionCallback* callback = nullptr;
	LinkedListElement<std::shared_ptr<Allocation>>* list_position = nullptr;
};

struct Page {
	char* ptr;
	std::shared_ptr<Allocation> current_allocation;
};

class PageCache {
private:
	std::mutex mutex;
	char* baseptr;

public:
	const size_t size, page_size;
	const unsigned n_pages;
	LinkedList<Page*> freePages;
	LinkedList<std::shared_ptr<Allocation>> lockedAllocations, unlockedAllocations;

	PageCache(char* baseptr, size_t total_size, size_t page_size) : size(total_size), baseptr(baseptr), page_size(page_size), n_pages(total_size/page_size) {
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
	bool trylock(std::shared_ptr<Allocation> allocation) {
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
	void lock(std::shared_ptr<Allocation> allocation) {
		CHECK(trylock(allocation)) << "Cannot lock evicted allocation";
	}

	void unlock(std::shared_ptr<Allocation> allocation) {
		std::lock_guard<std::mutex> lock(mutex);

		CHECK(!(allocation->evicted)) << "Tried unlocking an allocation that's already been evicted";

		if (--allocation->usage_count == 0) {
			// Unlock the allocation
			lockedAllocations.remove(allocation->list_position);
			allocation->list_position = unlockedAllocations.pushBack(allocation);
		}
	}

	std::shared_ptr<Allocation> alloc(unsigned n_pages, EvictionCallback* callback) {
		std::shared_ptr<Allocation> alloc = std::make_shared<Allocation>();
		alloc->callback = callback;
		alloc->pages.reserve(n_pages);

		std::lock_guard<std::mutex> lock(mutex);

		// Use up free pages
		while(alloc->pages.size() < n_pages) {
			if (!freePages.isEmpty()) {
				Page* p = freePages.popHead();
				p->current_allocation = alloc;
				alloc->pages.push_back(p);
			}
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
		}

		// Notify eviction handlers
		for (unsigned i = 0; i < callbacks.size(); i++) {
			callbacks[i]->evicted();
		}

		return alloc;
	}

	void free(std::shared_ptr<Allocation> allocation) {
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
	}

};




}
}

#endif