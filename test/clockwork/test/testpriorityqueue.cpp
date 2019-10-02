#include <catch2/catch.hpp>

#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "clockwork/clockworkruntime.h"

TEST_CASE("Priority Queue Simple Dequeue Order", "[queue]") {
    using namespace clockwork::clockworkruntime;

    time_release_priority_queue<int> q;

    std::vector<int*> elements;
    for (unsigned i = 0; i < 10; i++) {
        int* element = new int();
        q.enqueue(element, i);
        elements.push_back(element);
    }

    for (unsigned i = 0; i < 10; i++) {
        int* element = q.dequeue();
        REQUIRE(element == elements[i]);
    }
}

TEST_CASE("Priority Queue Reverse Dequeue Order", "[queue]") {
    using namespace clockwork::clockworkruntime;

    time_release_priority_queue<int> q;

    std::vector<int*> elements;
    for (unsigned i = 0; i < 10; i++) {
        int* element = new int();
        q.enqueue(element, 9-i);
        elements.push_back(element);
    }

    for (unsigned i = 0; i < 10; i++) {
        int* element = q.dequeue();
        REQUIRE(element == elements[9-i]);
    }
}

TEST_CASE("Priority Queue ZigZag Dequeue Order", "[queue]") {
    using namespace clockwork::clockworkruntime;

    time_release_priority_queue<int> q;


    std::vector<int> priorities = { 10, 0, 5, 8, 3, 7, 11, 1};
    std::unordered_map<int, int*> elems;

    for (int &priority : priorities) {
        int* element = new int();
        q.enqueue(element, priority);
        elems[priority] = element;
    }

    std::sort(priorities.begin(), priorities.end());

    for (int &priority : priorities) {
        int* element = q.dequeue();
        REQUIRE(element == elems[priority]);
    }
}

TEST_CASE("Priority Queue Multiple Identical Priorities", "[queue]") {
    using namespace clockwork::clockworkruntime;

    time_release_priority_queue<int> q;

    std::vector<int*> low;
    std::vector<int*> high;

    for (unsigned i = 0; i < 10; i++) {
        int* elow = new int();
        q.enqueue(elow, 10);
        low.push_back(elow);

        int* ehigh = new int();
        q.enqueue(ehigh, 20);
        high.push_back(ehigh);
    }

    for (unsigned i = 0; i < low.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(std::find(low.begin(), low.end(), e) != low.end());
        REQUIRE(std::find(high.begin(), high.end(), e) == high.end());
    }

    for (unsigned i = 0; i < high.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(std::find(low.begin(), low.end(), e) == low.end());
        REQUIRE(std::find(high.begin(), high.end(), e) != high.end());
    }
}

TEST_CASE("Priority Queue Eligible Time", "[queue]") {
    // using namespace clockwork;
    
    // size_t total_size = 100;
    // size_t page_size = 10;
    // void* baseptr = malloc(total_size);
    // bool allow_evictions = false;

    // PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size, allow_evictions);


    // std::shared_ptr<Allocation> alloc1 = cache->alloc(5, []{});
    // REQUIRE(alloc1 != nullptr);
    // cache->unlock(alloc1);

    // std::shared_ptr<Allocation> alloc2 = cache->alloc(3, []{});
    // REQUIRE(alloc2 != nullptr);
    // cache->unlock(alloc2);

    // std::shared_ptr<Allocation> alloc3 = cache->alloc(3, []{});
    // REQUIRE(alloc3 == nullptr);

    // std::shared_ptr<Allocation> alloc4 = cache->alloc(2, []{});
    // REQUIRE(alloc2 != nullptr);
    // cache->unlock(alloc4);

    // std::shared_ptr<Allocation> alloc5 = cache->alloc(1, []{});
    // REQUIRE(alloc5 == nullptr);

    // cache->free(alloc1);

    // std::shared_ptr<Allocation> alloc6 = cache->alloc(5, []{});
    // REQUIRE(alloc6 != nullptr);
}
