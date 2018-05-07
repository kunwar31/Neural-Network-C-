//
// Created by kunwar on 17/4/18.
//

#ifndef NEURALNETWORK_POKER_DATASET_H
#define NEURALNETWORK_POKER_DATASET_H


#include "bike_share_dataset.h"

class poker_dataset : public bike_share_dataset {
protected:
    void forward_pass(double* feature_batch, int n_features) override;
    void backpropogation(double* x,double* y,int n_features) override;
public:
    poker_dataset(int n1,int n2,int n3, float lr);
    double train(double features[][10],int n_features,int n_cols, double targets[][1], int n_targets);
    double* run(double* features, int n_features) override ;

    void relu_activation_function(double* x, int data_size);
    void clippedrelu_activation_function(double* x, int data_size,double clip);
};


#endif //NEURALNETWORK_POKER_DATASET_H
