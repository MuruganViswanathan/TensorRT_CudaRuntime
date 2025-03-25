#pragma once
#include <iostream>
#include <fstream>
#include <vector>


std::vector<float> preprocessImage(const std::string& imagePath);
void runInference(const std::string& modelPath, const std::string& imagePath);
//int query_main(int argc, char** argv)