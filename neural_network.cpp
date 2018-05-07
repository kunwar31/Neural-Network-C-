//
// Created by kunwar on 17/4/18.
//

#include <fstream>
#include "neural_network.h"
#include "iostream"


neural_network::neural_network(int n1,int n2,int n3,float lr)
{
    input_nodes = n1;
    hidden_nodes = n2;
    output_nodes = n3;
    learning_rate = lr;

    weights_input_to_hidden = new double * [input_nodes];
    weights_hidden_to_output = new double * [hidden_nodes];

    for (int i = 0; i < input_nodes; ++i) {
        weights_input_to_hidden[i] = new double[hidden_nodes];
    }
    for (int i = 0; i < hidden_nodes; ++i) {
        weights_hidden_to_output[i] = new double[output_nodes];
    }

}

void neural_network::activation_function(double* x, int data_size, int constant)
{
    // linear activation
    for (int i = 0; i < data_size; ++i) {

        x[i] *= constant;

    }
}

void neural_network::save_weights(std::string const &filename) {
    std::ofstream weights;
    weights.open("/home/kunwar/CLionProjects/NeuralNetwork/" + filename + ".data");
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            weights << weights_input_to_hidden[i][j] << "\n";

        }
    }
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            weights << weights_hidden_to_output[i][j] << "\n";
        }
    }
    weights.close();
}


void neural_network::load_weights(std::string const &filename) {
    std::ifstream weights;

    weights.open("/home/kunwar/CLionProjects/NeuralNetwork/" + filename + ".data");
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            std::string val;
            weights >> val;
            weights_input_to_hidden[i][j] = std::stod(val);

        }
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            std::string val;
            weights >> val;
            weights_hidden_to_output[i][j] = std::stod(val);
        }
    }
    weights.close();

}
