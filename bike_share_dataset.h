//
// Created by kunwar on 17/4/18.
//

#ifndef NEURALNETWORK_BIKE_SHARE_DATASET_H
#define NEURALNETWORK_BIKE_SHARE_DATASET_H


#include <random>
#include "neural_network.h"

class bike_share_dataset : public neural_network
{
protected:
    int n_records;

    double ** delta_weights_i_h;
    double ** delta_weights_h_o;
    double* hidden_outputs;
    double* final_outputs;
    void forward_pass(double* feature_batch, int n_features) override;
    void backpropogation(double* x,double* y,int n_features) override;
    void update_weights() override;
    void selu_activation_function(double* x, int data_size) ;
    void sigmoid_activation_function(double* x, int data_size) ;
    void leakyrelu_activation_function(double* x, int data_size) ;
    void refresh_delta_weights(bool first=false);
    void refresh_outputs(bool first=false);

public:
    bike_share_dataset(int n1,int n2,int n3,float lr);
    double train(double features[][56],int n_features,int n_cols, double targets[][1], int n_targets) override;
    double* run(double* features, int n_features) override;


};


#endif //NEURALNETWORK_BIKE_SHARE_DATASET_H
