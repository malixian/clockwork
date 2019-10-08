#include <catch2/catch.hpp>

#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "clockwork/util.h"
#include "clockwork/priority_queue.h"

TEST_CASE("Priority Queue Simple Dequeue Order", "[queue]") {
    using namespace clockwork;

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
    using namespace clockwork;

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
    using namespace clockwork;

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
    using namespace clockwork;

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
    using namespace clockwork;

    time_release_priority_queue<int> q;

    uint64_t now = clockwork::util::now();
    std::vector<int*> elements;
    std::vector<uint64_t> priorities;


    for (int i = -10; i < 10; i++) {
        int* e = new int();
        uint64_t priority = now + i * 200000000L; // 200ms * i
        q.enqueue(e, priority);
        elements.push_back(e);
        priorities.push_back(priority);
    }

    for (int i = 0; i < elements.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(e == elements[i]);
        REQUIRE(clockwork::util::now() >= priorities[i]);
    }
}
