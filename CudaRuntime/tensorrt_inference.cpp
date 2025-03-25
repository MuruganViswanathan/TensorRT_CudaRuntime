// Author: Murugan Viswanathan
// This project uses a pretrained ONNX file (GoogleNet model)
// Step 1: Install all dependent libraries: TensorRT, OpenCV etc onWindows
// Step 2: Use trtexec.exe to convert ONNX file to googlenet.engine file
//  command: trtexec.exe --onnx=googlenet_ultrasound.onnx --saveEngine=googlenet.engine --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:1x3x224x224 --fp16 --explicitBatch
// Step 3: Run the solution to infer a given US image file (hardcoded in kernel.cu). Example: ImageBreast.JPG
// This is a Prototype file for POC. Not a production level code.
// TODO: Functionally working , but to be refined

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include "tensorrt_inference.h"

using namespace std;
using namespace nvinfer1;


class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) std::cout << msg << std::endl;
    }
};

const int NUM_CLASSES = 7;
const int INPUT_H = 224;
const int INPUT_W = 224;
const int INPUT_C = 3;

// Class labels
std::vector<std::string> getClassLabels() {
    return { "Breast", "Kidney", "Liver", "Ovary", "Spleen", "Thyroid", "Uterus" };
}

// Preprocess image
std::vector<float> preprocessImage(const std::string& imagePath)
{
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "[ERROR] Image could not be loaded!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
    img.convertTo(img, CV_32FC3, 1.0 / 255);
    cv::subtract(img, cv::Scalar(0.485, 0.456, 0.406), img);
    cv::divide(img, cv::Scalar(0.229, 0.224, 0.225), img);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    std::vector<float> inputData(INPUT_C * INPUT_H * INPUT_W);
    for (int c = 0; c < INPUT_C; ++c) {
        std::memcpy(inputData.data() + c * INPUT_H * INPUT_W, channels[c].data, INPUT_H * INPUT_W * sizeof(float));
    }

    return inputData;
}

// Run inference
void runInference(const std::string& enginePath, const std::string& imagePath)
{
    Logger logger;
    IRuntime* runtime = createInferRuntime(logger);

    // Load serialized engine
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile.is_open()) {
        std::cerr << "[ERROR] Unable to open engine file!" << std::endl;
        return;
    }

    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    // Deserialize engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);
    if (!engine) {
        std::cerr << "[ERROR] Failed to load TensorRT engine!" << std::endl;
        return;
    }

    IExecutionContext* context = engine->createExecutionContext();

    // **NEW WAY TO GET INPUT & OUTPUT NAMES IN TENSORRT 10**
    const char* inputTensorName = context->getEngine().getIOTensorName(0);
    const char* outputTensorName = context->getEngine().getIOTensorName(1);

    // Get input shape dynamically
    nvinfer1::Dims inputDims = context->getTensorShape(inputTensorName);
    std::cout << "[INFO] Expected input shape: ";
    for (int i = 0; i < inputDims.nbDims; ++i) {
        std::cout << inputDims.d[i] << " ";
    }
    std::cout << std::endl;

    if (inputDims.nbDims != 4 || inputDims.d[0] != 1 || inputDims.d[1] != 3 || inputDims.d[2] != 224 || inputDims.d[3] != 224) {
        std::cerr << "[ERROR] Incorrect input shape! Expected (1,3,224,224)" << std::endl;
        return;
    }

    // Allocate GPU memory
    void* buffers[2];
    cudaMalloc(&buffers[0], INPUT_C * INPUT_H * INPUT_W * sizeof(float));  // Input buffer

    // Get output shape dynamically
    nvinfer1::Dims outputDims = context->getTensorShape(outputTensorName);
    int outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) outputSize *= static_cast<int>(outputDims.d[i]);

    cudaMalloc(&buffers[1], outputSize * sizeof(float));  // Output buffer

    // Preprocess image and copy to GPU
    std::vector<float> inputData = preprocessImage(imagePath);
    cudaMemcpy(buffers[0], inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Perform inference
    context->executeV2(buffers);

    // Retrieve output data
    std::vector<float> outputData(outputSize);
    cudaMemcpy(outputData.data(), buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    //// Get the top predicted class
    //auto maxIt = std::max_element(outputData.begin(), outputData.end());
    //int classId = std::distance(outputData.begin(), maxIt);
    //float confidence = *maxIt;

    // apply the softmax function to get valid probability values
    std::vector<float> probabilities(outputSize);
    float sumExp = 0.0f;
    for (int i = 0; i < outputSize; ++i) {
        probabilities[i] = std::exp(outputData[i]);
        sumExp += probabilities[i];
    }
    for (int i = 0; i < outputSize; ++i) {
        probabilities[i] /= sumExp;  // Normalize to get valid probabilities
    }

    // Get the top predicted class
    auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
    int classId = std::distance(probabilities.begin(), maxIt);
    float confidence = *maxIt;



    // Get class labels
    std::vector<std::string> classLabels = getClassLabels();
    std::string classLabel = (classId < classLabels.size()) ? classLabels[classId] : "Unknown";

    // Print results
    std::cout << "=================================================================================" << std::endl;
    std::cout << "    Predicted class: " << classLabel << " (ID: " << classId << "), Confidence: " << confidence * 100 << "%" << std::endl;
    std::cout << "=================================================================================" << std::endl;

    std::cout << "All class probabilities:" << std::endl;
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
        std::cout << classLabels[i] << ": " << probabilities[i] * 100 << "%" << std::endl;
    }

    // Free memory
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete engine;
    delete runtime;
}
