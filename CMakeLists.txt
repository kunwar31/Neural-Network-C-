cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetwork main.cpp neural_network.cpp neural_network.h bike_share_dataset.cpp bike_share_dataset.h poker_dataset.cpp poker_dataset.h matplotlibcpp.h Python.h)