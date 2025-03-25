#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include "tensorrt_inference.h"

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) std::cout << msg << std::endl;
    }
};

const int INPUT_H = 224;
const int INPUT_W = 224;
const int INPUT_C = 3;
const std::string INPUT_NAME = "input";
const std::string OUTPUT_NAME = "output";

std::vector<float> preprocessImage(const std::string& imagePath) 
{
    cv::Mat img = cv::imread(imagePath);
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
    img.convertTo(img, CV_32FC3, 1.0 / 255);
    std::vector<float> inputData(INPUT_C * INPUT_H * INPUT_W);
    for (int c = 0; c < INPUT_C; c++) {
        for (int i = 0; i < INPUT_H; i++) {
            for (int j = 0; j < INPUT_W; j++) {
                inputData[c * INPUT_H * INPUT_W + i * INPUT_W + j] = img.at<cv::Vec3f>(i, j)[c];
            }
        }
    }
    return inputData;
}



void runInference(const std::string& modelPath, const std::string& imagePath)
{
    Logger logger;
    IRuntime* runtime = createInferRuntime(logger);

    std::ifstream modelFile(modelPath, std::ios::binary);
    modelFile.seekg(0, std::ios::end);
    size_t modelSize = modelFile.tellg();
    modelFile.seekg(0, std::ios::beg);
    std::vector<char> modelData(modelSize);
    modelFile.read(modelData.data(), modelSize);
    modelFile.close();

    // Fix: Use correct deserializeCudaEngine() signature
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.data(), modelSize);

    // Create execution context
    IExecutionContext* context = engine->createExecutionContext();

    void* buffers[2];
    cudaMalloc(&buffers[0], INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[1], sizeof(float) * 1000);

    std::vector<float> inputData = preprocessImage(imagePath);
    cudaMemcpy(buffers[0], inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Fix: Use executeV2() instead of enqueueV2()
    context->executeV2(buffers);

    std::vector<float> outputData(1000);
    cudaMemcpy(outputData.data(), buffers[1], sizeof(float) * 1000, cudaMemcpyDeviceToHost);

    int classId = std::max_element(outputData.begin(), outputData.end()) - outputData.begin();
    std::cout << "Predicted class ID: " << classId << std::endl;

    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    delete context;
    delete engine;
    delete runtime;
}


