#include <unistd.h>
#include <libgen.h>
#include <fstream>

#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include <catch2/catch.hpp>

using namespace clockwork::model;

void assert_is_cool(Model* model) {
    int page_size = 16 * 1024 * 1024;
    char input[224*224*3*4];
    char output[1000*1*4];
    std::vector<char*> weights_pages;
    std::vector<char*> workspace_pages;

    REQUIRE_THROWS(model->uninstantiate_model_on_host());
    REQUIRE_THROWS(model->num_weights_pages(page_size));
    REQUIRE_THROWS(model->num_workspace_pages(page_size));
    REQUIRE_THROWS(model->unset_weights_pages());
    REQUIRE_THROWS(model->unset_workspace_pages());
    REQUIRE_THROWS(model->transfer_weights_to_device(NULL));
    REQUIRE_THROWS(model->input_size());
    REQUIRE_THROWS(model->output_size());
    REQUIRE_THROWS(model->transfer_input_to_device(input, NULL));
    REQUIRE_THROWS(model->transfer_output_from_device(output, NULL));
    REQUIRE_THROWS(model->call(NULL));

    REQUIRE_NOTHROW(model->set_weights_pages(weights_pages));
    REQUIRE_NOTHROW(model->set_workspace_pages(workspace_pages));
    REQUIRE_NOTHROW(model->unset_weights_pages());
    REQUIRE_NOTHROW(model->unset_workspace_pages());

    REQUIRE_THROWS(model->unset_weights_pages());
    REQUIRE_THROWS(model->unset_workspace_pages());
}

void assert_is_warm(Model* model) {
    int page_size = 16 * 1024 * 1024;
    int input_size = 224*224*3*4;
    int output_size = 1000 * 1 * 4;
    char input[input_size];
    char output[output_size];
    int num_weights_pages = 5;
    int num_workspace_pages = 2;
    std::vector<char*> weights_pages;
    std::vector<char*> workspace_pages;

    REQUIRE_THROWS(model->instantiate_model_on_host());
    REQUIRE_THROWS(model->unset_weights_pages());
    REQUIRE_THROWS(model->unset_workspace_pages());
    REQUIRE_THROWS(model->transfer_weights_to_device(NULL));
    REQUIRE_THROWS(model->transfer_input_to_device(input, NULL));
    REQUIRE_THROWS(model->transfer_output_from_device(output, NULL));
    REQUIRE_THROWS(model->call(NULL));

    REQUIRE(num_weights_pages == model->num_weights_pages(page_size));
    REQUIRE(num_workspace_pages == model->num_workspace_pages(page_size));
    REQUIRE(input_size == model->input_size());
    REQUIRE(output_size == model->output_size());

    REQUIRE_NOTHROW(model->set_weights_pages(weights_pages));
    REQUIRE_NOTHROW(model->set_workspace_pages(workspace_pages));
    REQUIRE_NOTHROW(model->unset_weights_pages());
    REQUIRE_NOTHROW(model->unset_workspace_pages());
    
    REQUIRE_THROWS(model->unset_weights_pages());
}

void assert_is_hot(Model* model) {
    int page_size = 16 * 1024 * 1024;
    int input_size = 224*224*3*4;
    int output_size = 1000 * 1 * 4;
    char input[input_size];
    char output[output_size];
    int num_weights_pages = 5;
    int num_workspace_pages = 2;
    std::vector<char*> weights_pages = make_cuda_pages(page_size, num_weights_pages);
    std::vector<char*> workspace_pages = make_cuda_pages(page_size, num_workspace_pages);

    REQUIRE_THROWS(model->instantiate_model_on_host());
    REQUIRE_THROWS(model->unset_weights_pages());
    REQUIRE_THROWS(model->unset_workspace_pages());
    REQUIRE_THROWS(model->transfer_weights_to_device(NULL));
    REQUIRE_THROWS(model->transfer_input_to_device(input, NULL));
    REQUIRE_THROWS(model->transfer_output_from_device(output, NULL));
    REQUIRE_THROWS(model->call(NULL));

    REQUIRE(num_weights_pages == model->num_weights_pages(page_size));
    REQUIRE(num_workspace_pages == model->num_workspace_pages(page_size));
    REQUIRE(input_size == model->input_size());
    REQUIRE(output_size == model->output_size());

    REQUIRE_NOTHROW(model->set_weights_pages(weights_pages));
    REQUIRE_NOTHROW(model->set_workspace_pages(workspace_pages));
    REQUIRE_NOTHROW(model->transfer_weights_to_device(NULL));
    REQUIRE_NOTHROW(model->transfer_input_to_device(input, NULL));
    REQUIRE_NOTHROW(model->transfer_output_from_device(output, NULL));
    REQUIRE_NOTHROW(model->call(NULL));

    cuda_synchronize(NULL);

    REQUIRE_NOTHROW(model->unset_weights_pages());
    REQUIRE_NOTHROW(model->unset_workspace_pages());

    free_cuda_pages(weights_pages);
    free_cuda_pages(workspace_pages);
}

TEST_CASE("Load model from disk", "[model]") {

    std::string f = clockwork::util::get_example_model();

    Model* model = nullptr;
    REQUIRE_NOTHROW(model = Model::loadFromDisk(f+".so", f+".clockwork", f+".clockwork_params"));
    REQUIRE(model != nullptr);
    delete model;

}

TEST_CASE("Model lifecycle 1", "[model]") {

    std::string f = clockwork::util::get_example_model();

    Model* model = nullptr;
    REQUIRE_NOTHROW(model = Model::loadFromDisk(f+".so", f+".clockwork", f+".clockwork_params"));
    REQUIRE(model != nullptr);

    int page_size;

    REQUIRE_THROWS(model->uninstantiate_model_on_host());

    REQUIRE_THROWS(model->num_weights_pages(page_size));
    REQUIRE_THROWS(model->num_weights_pages(page_size));

    REQUIRE_NOTHROW(model->instantiate_model_on_host());
    REQUIRE_NOTHROW(model->uninstantiate_model_on_host());
    REQUIRE_NOTHROW(model->instantiate_model_on_host());
    REQUIRE_NOTHROW(model->uninstantiate_model_on_host());
    REQUIRE_THROWS(model->uninstantiate_model_on_host());

    delete model;
}


TEST_CASE("Model Lifecycle 2", "[model]") {

    std::string f = clockwork::util::get_example_model();

    Model* model = nullptr;
    REQUIRE_NOTHROW(model = Model::loadFromDisk(f+".so", f+".clockwork", f+".clockwork_params"));
    REQUIRE(model != nullptr);
    
    assert_is_cool(model);

    model->instantiate_model_on_host();

    assert_is_warm(model);

    model->instantiate_model_on_device();

    assert_is_hot(model);

    model->uninstantiate_model_on_device();

    assert_is_warm(model);

    model->instantiate_model_on_device();

    assert_is_hot(model);

    model->uninstantiate_model_on_device();

    assert_is_warm(model);

    model->uninstantiate_model_on_host();

    assert_is_cool(model);

    model->instantiate_model_on_host();

    assert_is_warm(model);

    model->instantiate_model_on_device();

    assert_is_hot(model);


    delete model;

}

TEST_CASE("Model produces correct output", "[model]") {

    int page_size = 16 * 1024 * 1024;
    int input_size = 224*224*3*4;
    int output_size = 1000 * 1 * 4;
    int num_weights_pages = 5;
    int num_workspace_pages = 2;
    std::vector<char*> weights_pages = make_cuda_pages(page_size, num_weights_pages);
    std::vector<char*> workspace_pages = make_cuda_pages(page_size, num_workspace_pages);

    std::string f = clockwork::util::get_example_model();

    Model* model = Model::loadFromDisk(f+".so", f+".clockwork", f+".clockwork_params");
    
    model->instantiate_model_on_host();
    model->instantiate_model_on_device();
    model->set_weights_pages(weights_pages);
    model->set_workspace_pages(workspace_pages);
    model->transfer_weights_to_device(NULL);

    std::ifstream in(f+".input");
    std::string input_filename = f+".input";
    std::string output_filename = f+".output";
    std::string input, expectedOutput;
    char actualOutput[output_size];
    clockwork::util::readFileAsString(input_filename, input);
    clockwork::util::readFileAsString(output_filename, expectedOutput);

    REQUIRE(input.size() == input_size);
    model->transfer_input_to_device(input.data(), NULL);
    model->call(NULL);
    model->transfer_output_from_device(actualOutput, NULL);

    cuda_synchronize(NULL);

    for (unsigned i = 0; i < output_size; i++) {
        REQUIRE(actualOutput[i] == expectedOutput[i]);
    }
}