#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

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
    REQUIRE_THROWS(model->input_size());
    REQUIRE_THROWS(model->output_size());
    REQUIRE_THROWS(model->call(weights_pages, workspace_pages, NULL));
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
    REQUIRE_THROWS(model->call(weights_pages, workspace_pages, NULL));

    REQUIRE(num_weights_pages == model->num_weights_pages(page_size));
    REQUIRE(num_workspace_pages == model->num_workspace_pages(page_size));
    REQUIRE(input_size == model->input_size());
    REQUIRE(output_size == model->output_size());
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

    REQUIRE(num_weights_pages == model->num_weights_pages(page_size));
    REQUIRE(num_workspace_pages == model->num_workspace_pages(page_size));
    REQUIRE(input_size == model->input_size());
    REQUIRE(output_size == model->output_size());

    REQUIRE_NOTHROW(model->transfer_weights_to_device(weights_pages, NULL));
    REQUIRE_NOTHROW(model->transfer_input_to_device(input, workspace_pages, NULL));
    REQUIRE_NOTHROW(model->transfer_output_from_device(output, workspace_pages, NULL));
    REQUIRE_NOTHROW(model->call(weights_pages, workspace_pages, NULL));

    cuda_synchronize(NULL);

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

TEST_CASE("Model produces correct output", "[e2e]") {

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
    model->transfer_weights_to_device(weights_pages, NULL);

    std::ifstream in(f+".input");
    std::string input_filename = f+".input";
    std::string output_filename = f+".output";
    std::string input, expectedOutput;
    char actualOutput[output_size];
    clockwork::util::readFileAsString(input_filename, input);
    clockwork::util::readFileAsString(output_filename, expectedOutput);

    REQUIRE(input.size() == input_size);
    model->transfer_input_to_device(input.data(), workspace_pages, NULL);
    model->call(weights_pages, workspace_pages, NULL);
    model->transfer_output_from_device(actualOutput, workspace_pages, NULL);

    REQUIRE(output_size == expectedOutput.size());

    cuda_synchronize(NULL);

    float* actualOutputF = static_cast<float*>(static_cast<void*>(actualOutput));
    const float* expectedOutputF = reinterpret_cast<const float*>(expectedOutput.data());

    auto max_index_actual = std::distance(actualOutputF, std::max_element(actualOutputF, actualOutputF + 1000));
    auto max_index_expect = std::distance(expectedOutputF, std::max_element(expectedOutputF, expectedOutputF + 1000));
    REQUIRE(max_index_expect == max_index_actual);

    for (unsigned i = 0; i < output_size/4; i++) {
        REQUIRE(actualOutputF[i] == expectedOutputF[i]);
    }
}

TEST_CASE("Batched model produces correct output", "[e2e2]") {

    int page_size = 16 * 1024 * 1024;
    int input_size = 2*224*224*3*4;
    int output_size = 1000 * 2 * 4;

    std::string f = clockwork::util::get_example_model("resnet18_tesla-m40_batchsize2");

    Model* model = Model::loadFromDisk(f+".so", f+".clockwork", f+".clockwork_params");
    
    model->instantiate_model_on_host();
    model->instantiate_model_on_device();

    std::vector<char*> weights_pages = make_cuda_pages(page_size, model->num_weights_pages(page_size));
    model->transfer_weights_to_device(weights_pages, NULL);



    std::string input, expectedOutput;
    clockwork::util::readFileAsString(f+".input", input);
    clockwork::util::readFileAsString(f+".output", expectedOutput);

    REQUIRE(input_size == 2 * input.size());
    REQUIRE(output_size == 2 * expectedOutput.size());

    char batched_input[input_size];
    std::memcpy(batched_input, input.data(), input.size());
    std::memcpy(batched_input + input.size(), input.data(), input.size());

    char batched_expected_output[output_size];
    std::memcpy(batched_expected_output, expectedOutput.data(), expectedOutput.size());
    std::memcpy(batched_expected_output + expectedOutput.size(), expectedOutput.data(), expectedOutput.size());


    std::vector<char*> workspace_pages = make_cuda_pages(page_size, model->num_workspace_pages(page_size));
    model->transfer_input_to_device(batched_input, workspace_pages, NULL);
    model->call(weights_pages, workspace_pages, NULL);

    char actualOutput[output_size];
    model->transfer_output_from_device(actualOutput, workspace_pages, NULL);

    cuda_synchronize(NULL);

    float* actualOutputF = static_cast<float*>(static_cast<void*>(actualOutput));
    float* expectedOutputF = static_cast<float*>(static_cast<void*>(batched_expected_output));

    auto max_index_actual_1 = std::distance(actualOutputF, std::max_element(actualOutputF, actualOutputF + 1000));
    auto max_index_expect_1 = std::distance(expectedOutputF, std::max_element(expectedOutputF, expectedOutputF + 1000));
    REQUIRE(max_index_expect_1 == max_index_actual_1);

    std::cout << "max_index_actual_1 is " << max_index_actual_1 << std::endl;

    auto max_index_actual_2 = std::distance(actualOutputF+1000, std::max_element(actualOutputF+1000, actualOutputF + 2000));
    auto max_index_expect_2 = std::distance(expectedOutputF+1000, std::max_element(expectedOutputF+1000, expectedOutputF + 2000));
    REQUIRE(max_index_expect_2 == max_index_actual_2);

    std::cout << "max_index_actual_2 is " << max_index_actual_2 << std::endl;

    for (unsigned i = 0; i < output_size/4; i++) {
        REQUIRE(actualOutputF[i] == expectedOutputF[i]);
    }
}