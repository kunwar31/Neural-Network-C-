//
// Created by kunwar on 17/4/18.
//


#include "bike_share_dataset.h"

bike_share_dataset::bike_share_dataset(int n1, int n2, int n3, float lr): neural_network(n1, n2, n3, lr) {
    std::normal_distribution < double > i_t_h(0.0, pow(input_nodes, -0.5));
    std::normal_distribution < double > h_t_o(0.0, pow(hidden_nodes, -0.5));
    std::default_random_engine generator;

    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            double number = i_t_h(generator);
            weights_input_to_hidden[i][j] = number;
            //std::cout<< weights_input_to_hidden[i][j] << '\n';
        }
    }
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            double number = h_t_o(generator);
            weights_hidden_to_output[i][j] = number;
            //std::cout<< weights_hidden_to_output[i][j] << '\n';
        }
    }
    refresh_delta_weights(true);
    refresh_outputs(true);

}

void bike_share_dataset::refresh_delta_weights(bool first) {
    if (first) {
        delta_weights_i_h = new double * [input_nodes];
        delta_weights_h_o = new double * [hidden_nodes];
    }
    for (int i = 0; i < input_nodes; ++i) {
        if (first)
            delta_weights_i_h[i] = new double[hidden_nodes];
        for (int j = 0; j < hidden_nodes; ++j) {
            delta_weights_i_h[i][j] = 0;
        }
    }
    for (int i = 0; i < hidden_nodes; ++i) {
        if (first)
            delta_weights_h_o[i] = new double[output_nodes];
        for (int j = 0; j < output_nodes; ++j) {
            delta_weights_h_o[i][j] = 0;
        }
    }

}

void bike_share_dataset::refresh_outputs(bool first) {
    if (first) {
        hidden_outputs = new double[hidden_nodes];
        final_outputs = new double[output_nodes];
    }
    for (int i = 0; i < hidden_nodes; ++i) {
        hidden_outputs[i] = 0;
    }
    for (int i = 0; i < output_nodes; ++i) {
        final_outputs[i] = 0;
    }
}

void bike_share_dataset::selu_activation_function(double * x, int data_size) {

    double lambda = 1.050700987355;
    double alpha = 1.6732632423543;
    for (int i = 0; i < data_size; ++i) {
        if (x[i]<=0)
        {
            x[i] = lambda * (exp(x[i])*alpha - alpha);
        }
    }

}

void bike_share_dataset::sigmoid_activation_function(double * x, int data_size) {

    for (int i = 0; i < data_size; ++i) {

        x[i] = 1 / (1 + exp(-x[i]));

    }

}

void bike_share_dataset::leakyrelu_activation_function(double * x, int data_size) {

    for (int i = 0; i < data_size; ++i) {

        x[i] = std::max(0.15*x[i],x[i]);

    }

}


double bike_share_dataset::train(double features[][56], int n_features, int n_cols, double targets[][1], int n_targets) {
    n_records = n_cols;

    auto feature_batch = new double[n_features];
    auto target_batch = new double[n_targets];
    double error_sum_squared = 0;
    refresh_delta_weights();

    for (int i = 0; i < n_records; ++i) {

        for (int j = 0; j < n_features; ++j) {
            feature_batch[j] = features[i][j];
        }
        for (int j = 0; j < n_targets; ++j) {
            target_batch[j] = targets[i][j];
        }

        forward_pass(feature_batch, n_features);
        error_sum_squared += pow(targets[i][0] - final_outputs[0], 2);
        //std::cout << final_outputs[0] << std::endl;
        backpropogation(feature_batch, target_batch, n_features);
    }
    update_weights();
    delete feature_batch;
    delete target_batch;
    return error_sum_squared / n_records;

}
void bike_share_dataset::forward_pass(double * feature_batch, int n_features) {
    refresh_outputs();
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            hidden_outputs[j] += feature_batch[i] * weights_input_to_hidden[i][j];
            if (weights_input_to_hidden[i][j] > 5 or weights_input_to_hidden[i][j] < -5) {
                //std::cout<< weights_input_to_hidden[i][j] << " i:" << i << " j:" << j << "\n";
            }
        }
    }
    leakyrelu_activation_function(hidden_outputs, hidden_nodes);

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            final_outputs[j] += hidden_outputs[i] * weights_hidden_to_output[i][j];
        }
    }
    leakyrelu_activation_function(final_outputs, output_nodes);
}
void bike_share_dataset::backpropogation(double * x, double * y, int n_features) {
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
            output_error_term[i] = output_error * 0.15 * 0.15;
        }
    }

    for (int i = 0; i < hidden_nodes * output_nodes; ++i)
        hidden_error += output_error_term[0] * * ( & weights_hidden_to_output[0][0] + i);

    for (int i = 0; i < hidden_nodes; ++i)    {
        if (hidden_outputs[i]>0)
        hidden_error_term[i] = hidden_error;
        else
        {
            hidden_error_term[i] = hidden_error * 0.15 * 0.15;
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
void bike_share_dataset::update_weights() {
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            weights_hidden_to_output[i][j] += learning_rate * delta_weights_h_o[i][j] / n_records;
            //std::cout<< weights_hidden_to_output[i][j] << std::endl;
        }
    }

    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            weights_input_to_hidden[i][j] += learning_rate * delta_weights_i_h[i][j] / n_records;
            //std::cout<< weights_input_to_hidden[i][j] << std::endl;
        }
    }
}
double * bike_share_dataset::run(double * features, int n_features) {
    refresh_outputs();
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            hidden_outputs[j] += features[i] * weights_input_to_hidden[i][j];
        }
    }
    leakyrelu_activation_function(hidden_outputs, hidden_nodes);

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            final_outputs[j] += hidden_outputs[i] * weights_hidden_to_output[i][j];
        }
    }
    leakyrelu_activation_function(final_outputs, output_nodes);
    return final_outputs;
}