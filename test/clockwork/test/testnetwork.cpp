#include <catch2/catch.hpp>
#include "clockwork/worker.h"
#include "clockwork/network/worker.h"


using namespace clockwork;

TEST_CASE("Test Worker Server Shutdown", "[network]") {

    auto clockwork = new clockwork::ClockworkWorker();
    auto server = new clockwork::network::worker::Server(clockwork);
    clockwork->controller = server;

    clockwork->shutdown(false);
    server->shutdown(false);

    clockwork->join();
    server->join();
}