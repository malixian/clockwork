// 020-TestCase-2.cpp

// main() provided by Catch in file 020-TestCase-1.cpp.

#include <catch2/catch.hpp>

#include <cstdlib>

#include "clockwork/model/memory.h"


int Factorial( int number ) {
   return number <= 1 ? number : Factorial( number - 1 ) * number;  // fail
// return number <= 1 ? 1      : Factorial( number - 1 ) * number;  // pass
}

TEST_CASE( "2: Factorial of 0 is 1 (fail)", "[multi-file:2]" ) {
    REQUIRE( Factorial(0) == 1 );
}

TEST_CASE( "2: Factorials of 1 and higher are computed (pass)", "[multi-file:2]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}

TEST_CASE("Create Page Cache with bad sizes", "[memory]") {

    using namespace clockwork::model;
    
    size_t total_size = 100;
    size_t page_size = 11;
    void* baseptr = malloc(total_size);
    
    PageCache* cache;
    REQUIRE_THROWS(cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size));
}

TEST_CASE("Simple Page Cache with one page", "[memory]") {

    using namespace clockwork::model;
    
    size_t total_size = 100;
    size_t page_size = 100;
    void* baseptr = malloc(total_size);
    
    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size);

    REQUIRE( cache->alloc(1, nullptr) != nullptr );
    REQUIRE( cache->alloc(1, nullptr) == nullptr );
}

// Compile: see 020-TestCase-1.cpp

// Expected compact output (all assertions):
//
// prompt> 020-TestCase --reporter compact --success
// 020-TestCase-2.cpp:13: failed: Factorial(0) == 1 for: 0 == 1
// 020-TestCase-2.cpp:17: passed: Factorial(1) == 1 for: 1 == 1
// 020-TestCase-2.cpp:18: passed: Factorial(2) == 2 for: 2 == 2
// 020-TestCase-2.cpp:19: passed: Factorial(3) == 6 for: 6 == 6
// 020-TestCase-2.cpp:20: passed: Factorial(10) == 3628800 for: 3628800 (0x375f00) == 3628800 (0x375f00)
// Failed 1 test case, failed 1 assertion.