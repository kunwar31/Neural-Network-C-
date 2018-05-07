//
// Created by kunwar on 17/4/18.
//

#ifndef NEURALNETWORK_NEURAL_NETWORK_H
#define NEURALNETWORK_NEURAL_NETWORK_H

#include "string"

class neural_network
    {
    protected:
        int hidden_nodes;
        int input_nodes;
        int output_nodes;
        float learning_rate;
        double ** weights_input_to_hidden;
        double ** weights_hidden_to_output;
        virtual void forward_pass(double* feature_batch,int n_features) = 0;
        virtual void backpropogation(double* x,double* y,int n_features) = 0;
        virtual void update_weights() = 0;
        virtual void activation_function(double* x, int data_size, int constant);
    public:
        neural_network(int n1,int n2,int n3,float lr);
        virtual double train(double features[][56],int n_features,int n_cols, double targets[][1], int n_targets) = 0;
        virtual double* run(double* features, int n_features) = 0;
        void save_weights(std::string const &filename);
        void load_weights(std::string const &filename);
};


#endif //NEURALNETWORK_NEURAL_NETWORK_H
