#include <random>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include "bike_share_dataset.h"
#include "poker_dataset.h"
#include "matplotlibcpp.h"

void plot(std::vector <double> targets,std::vector <double> old_predicts,std::vector <double> new_predicts,double mean,double std)
{
    for (int i = 0;i<targets.size();++i)
    {
        targets[i] *= std;
        targets[i] += mean;
        old_predicts[i] *= std;
        old_predicts[i] += mean;
        new_predicts[i] *= std;
        new_predicts[i] += mean;
    }

    //matplotlibcpp::plot(targets,predicts);
    matplotlibcpp::named_plot("Targets",targets);
    matplotlibcpp::named_plot("Random Predictions",old_predicts);
    matplotlibcpp::named_plot("Predictions after training",new_predicts);
    matplotlibcpp::legend();
    matplotlibcpp::show();
}


void bike_share_test(int hidden_nodes,float lr,int epochs,bool load_weights,int MAX_ENTRIES=10000)
{
    std::ifstream file ("/home/kunwar/CLionProjects/NeuralNetwork/new_data.txt", std::ifstream::in);
    std::string value;
    double features[MAX_ENTRIES][56];
    double targets[MAX_ENTRIES][1];
    int i = 0;

    while ( file.good() and i < MAX_ENTRIES)
    {
        std::getline ( file, value, '\n');
        if (i==0)
        {
            ++i;
            continue;
        }
        int f_c = 0;
        int j = 0;
        std::string substr;
        while (f_c < value.length())
        {
            if (value[f_c]==',')
            {
                ++j;
                if (j>2 and j<=58){
                    features[i-1][j-3] = std::stod(substr);
                }
                substr = "";
            }
            else if (j>=2)
            {
                substr += value[f_c];
            }
            ++f_c;

        }
        targets[i-1][0] = std::stod(substr);
        ++i;
    }

    std::cout << "\nLoaded dataset with " << i << " samples.\n";

    std::vector <double> targets_vec ;
    for (int a =0;a<250;++a)
    {
        targets_vec.push_back(targets[a][0]);
    }

    auto b1 = new bike_share_dataset(56,hidden_nodes,1,lr);

    std::vector <double> old_predicts_vec ;
    for (int a =0;a<250;++a)
    {
        old_predicts_vec.push_back(b1->run(&features[a][0], 56)[0]);
    }


    if (load_weights)
    {
        b1->load_weights("result1");
    }
    else
    {
        for (int j = 0;j<epochs;++j)
            std::cout << "Train Loss: " << b1->train(features, 56,i-1,targets,1) << " after " << j+1 << " iterations.\r"<< std::flush;
    }

    std::vector <double> new_predicts_vec ;
    for (int a =0;a<250;++a)
    {
        new_predicts_vec.push_back(b1->run(&features[a][0], 56)[0]);
    }

    plot(targets_vec,old_predicts_vec,new_predicts_vec,189.46308763450142,181.38759909186527);
    b1->save_weights("result1");
}

void poker_test(int hidden_nodes,float lr, int epochs,bool load_weights,int MAX_ENTRIES = 1000)
{
    std::ifstream file ("/home/kunwar/CLionProjects/NeuralNetwork/poker-hand.data", std::ifstream::in); // declare file stream
    std::string value;
    double features[MAX_ENTRIES][10];
    double targets[MAX_ENTRIES][1];
    int i = 0;

    while ( file.good() and i < MAX_ENTRIES-1)
    {
        std::getline ( file, value, '\n');
        int f_c = 0;
        int j = 0;
        std::string substr;
        while (f_c < value.length())
        {
            if (value[f_c]==',')
            {
                ++j;
                if (j>2 and j<=12){
                    features[i][j-3] = std::stod(substr);
                }
                substr = "";
            }
            else if (j>=2)
            {
                substr += value[f_c];
            }
            ++f_c;

        }
        targets[i][0] = std::stod(substr);
        ++i;
    }

    std::cout << "\nLoaded dataset with " << i+1 << " samples.\n";
    std::vector <double> targets_vec ;
    for (int a =0;a<250;++a)
    {
        targets_vec.push_back(targets[a][0]);
    }

    auto b1 = new poker_dataset(10,hidden_nodes,1,lr);

    std::vector <double> old_predicts_vec ;
    for (int a =0;a<250;++a)
    {
        old_predicts_vec.push_back(b1->run(&features[a][0], 56)[0]);
    }
    if (load_weights)
    {
        b1->load_weights("result2");
    }
    else
    {
        for (int j = 0;j<epochs;++j)
            std::cout << "Train Loss: " << b1->train(features, 10,i,targets,1) << " after " << j+1 << " iterations.\r"<< std::flush;
    }


    std::vector <double> new_predicts_vec ;
    for (int a =0;a<250;++a)
    {
        new_predicts_vec.push_back(b1->run(&features[a][0], 56)[0]);
    }

    plot(targets_vec,old_predicts_vec,new_predicts_vec,0,1);
    b1->save_weights("result2");

}

int main()
{
    //bike_share_test(10,0.6,3000,false,18000);
    poker_test(3,0.01,1000000,false,5000);
    //bike_share_test(20,0.112,300,true,18000);
    //poker_test(7,0.5,10000,true,25000);
    return 0;
}