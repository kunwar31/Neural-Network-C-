//
// Created by kunwar on 17/4/18.
//

#include <iostream>
#include "poker_dataset.h"

void poker_dataset::relu_activation_function(double* x, int data_size)
{
    for (int i = 0; i < data_size; ++i) {
        x[i] = std::max(0.0,x[i]);
    }
}

void poker_dataset::clippedrelu_activation_function(double* x, int data_size,double clip)
{
    for (int i = 0; i < data_size; ++i) {
        x[i] = std::max(0.0,x[i]);
        x[i] = std::min(clip,x[i]);
    }
}

void poker_dataset::forward_pass(double *feature_batch, int n_features) {
    refresh_outputs();
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            hidden_outputs[j] += feature_batch[i] * weights_input_to_hidden[i][j];
        }
    }
    relu_activation_function(hidden_outputs,hidden_nodes);

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            final_outputs[j] += hidden_outputs[i] * weights_hidden_to_output[i][j];
        }
    }

    relu_activation_function(final_outputs, output_nodes);

}

poker_dataset::poker_dataset(int n1, int n2, int n3, float lr) : bike_share_dataset(n1,n2,n3,lr){
    std::normal_distribution < double > i_t_h(0, pow(input_nodes, -0.5));
    std::normal_distribution < double > h_t_o(0, pow(hidden_nodes, -0.5));
    std::default_random_engine generator;

    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            double number = i_t_h(generator);
            weights_input_to_hidden[i][j] = number;
        }
    }
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            double number = h_t_o(generator);
            weights_hidden_to_output[i][j] = number;
        }
    }

}



double poker_dataset::train(double (*features)[10], int n_features, int n_cols, double (*targets)[1], int n_targets) {
    bike_share_dataset::n_records = n_cols;
    auto feature_batch = new double[n_features];
    auto target_batch = new double[n_targets];
    double error_sum_squared = 0;
    bike_share_dataset::refresh_delta_weights();
    for (int i = 0; i < bike_share_dataset::n_records; ++i) {

        for (int j = 0; j < n_features; ++j) {
            feature_batch[j] = features[i][j];
        }
        for (int j = 0; j < n_targets; ++j) {
            target_batch[j] = targets[i][j];
        }
        poker_dataset::forward_pass(feature_batch, n_features);
        error_sum_squared += pow(targets[i][0] - final_outputs[0], 2);
        bike_share_dataset::backpropogation(feature_batch, target_batch, n_features);

    }
    bike_share_dataset::update_weights();
    delete feature_batch;
    delete target_batch;
    return error_sum_squared / n_records;
}


double *poker_dataset::run(double *features, int n_features) {
    refresh_outputs();
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            hidden_outputs[j] += features[i] * weights_input_to_hidden[i][j];

        }
    }
    relu_activation_function(hidden_outputs,hidden_nodes);

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            final_outputs[j] += hidden_outputs[i] * weights_hidden_to_output[i][j];
        }
    }

    relu_activation_function(final_outputs, output_nodes);
    return final_outputs;

}

void poker_dataset::backpropogation(double *x, double *y, int n_features) {
    auto output_error_term = new double[output_nodes];
    double hidden_error = 0;
    double output_error = 0;
    auto hidden_error_term = new double[hidden_nodes];
    for (int i = 0; i < output_nodes; ++i)
        output_error += y[i] - final_outputs[i];

    for (int i = 0; i < output_nodes; ++i)    {
        if (final_outputs[i]>0)
            output_error_term[i] = output_error;
        else
        {
            output_error_term[i] = 0;
        }
    }

    for (int i = 0; i < hidden_nodes * output_nodes; ++i)
        hidden_error += output_error_term[0] * * ( & weights_hidden_to_output[0][0] + i);

    for (int i = 0; i < hidden_nodes; ++i)    {
        if (hidden_outputs[i]>0)
            hidden_error_term[i] = hidden_error;
        else
        {
            hidden_error_term[i] = 0;
        }
    }

    // UPDATE DELTA OF WEIGHTS
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            delta_weights_h_o[i][j] += output_error_term[j] * hidden_outputs[i];
            //std::cout << learning_rate *  output_error_term[j] * hidden_outputs[i] << std::endl;
        }
    }
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            delta_weights_i_h[i][j] += hidden_error_term[j] * x[i];
            //std::cout << learning_rate * hidden_error_term[j] * x[i] << std::endl;
        }
    }
    delete output_error_term;
    delete hidden_error_term;
}
