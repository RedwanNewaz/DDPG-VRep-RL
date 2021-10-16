#ifndef VREP_EXT_PLUGIN_SKELETON_NG_H_INCLUDED
#define VREP_EXT_PLUGIN_SKELETON_NG_H_INCLUDED
#include <iostream>
#include <fstream>
#include "stubs.h"

#include "ddpg_plugin/keras_model.h"
#include "ddpg_plugin/generate_sample.h"
#include "motiongeneration.h"
#include "TrajGenerator/RapidTrajectoryGenerator.h"



using namespace std;
using namespace RapidQuadrocopterTrajectoryGenerator;



string inline read_param(){

    string weight   ;
// Read from the text file
    std::ifstream infile("rl_param.txt");
    infile >> weight;
    cout << "[+] loading weight from " << weight << endl;
    return weight;

}

vector<double> inline linspace(double a, double b, int n) {
    vector<double> array;
    double step = (b-a) / (n-1);
    
    while(a <= b) {
        array.push_back(a);
        a += step;           // could recode to better handle rounding errors
    }
    return array;
}
#endif // VREP_EXT_PLUGIN_SKELETON_NG_H_INCLUDED
