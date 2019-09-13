#ifndef _CLOCKWORK_MODEL_MEMORY_H_
#define _CLOCKWORK_MODEL_MEMORY_H_

#include <mutex>
#include <memory>
#include <atomic>
#include <vector>

namespace clockwork {

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
public:
	LinkedListElement<T>* head = nullptr;
	LinkedListElement<T>* tail = nullptr;

	bool isEmpty() {
		return head==nullptr;
	}

	T popHead() {
		if (isEmpty()) return nullptr;
		LinkedListElement<T>* elem = head;
		if (elem != nullptr) {
			head = head->next;
			if (head != nullptr) head->prev = nullptr;
		}
		if (head == nullptr && tail == elem) tail = nullptr;
		T data = elem->data;
		delete elem;
		return data;
	}

	T popTail() {
		if (isEmpty()) return nullptr;
		LinkedListElement<T>* elem = tail;
		if (elem != nullptr) {
			tail = tail->prev;
			if (tail != nullptr) tail->next = nullptr;
		}
		if (tail == nullptr && head == elem) head = nullptr;
		T data = elem->data;
		delete elem;
		return data;
	}

	bool remove(LinkedListElement<T>* element) {
		if (element == nullptr || element->container != this) return false;
		if (element->next != nullptr) element->next->prev = element->prev;
		else if (tail == element) tail = element->prev;
		if (element->prev != nullptr) element->prev->next = element->next;
		else if (head == element) head = element->next;
		delete element;
		return true;
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

	PageCache(char* baseptr, size_t total_size, size_t page_size);

	/* 
	Locks the allocation if it hasn't been evicted
	*/
	bool trylock(std::shared_ptr<Allocation> allocation);

	/* 
	Locks the allocation; error if it's evicted
	*/
	void lock(std::shared_ptr<Allocation> allocation);
	void unlock(std::shared_ptr<Allocation> allocation);

	/*
	Alloc will also lock the allocation immediately
	*/
	std::shared_ptr<Allocation> alloc(unsigned n_pages, EvictionCallback* callback);
	void free(std::shared_ptr<Allocation> allocation);

};

}

#endif