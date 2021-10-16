#ifndef GENERATE_SAMPLE_H
#define GENERATE_SAMPLE_H
#define STATE_SIZE 12
#include <iostream>
#include <vector>
#include "keras_model.h"
using namespace std;
using namespace keras;

inline void printVec(vector<float>& path){
    /* print the 1D input vector which has 12 elements
     */
    for(int i=0; i<path.size(); ++i)
      std::cout << path[i] << ' ';
    cout<<endl;
}

class generate_sample
{
public:
    generate_sample();
    void update_robot(vector<float>& x_t);
    void update_target(vector<float>& x_t);
    std::shared_ptr<DataChunk> get_sample();

private:
    vector <float> __robot_state, __target_state;
//    DataChunk sample = new DataChunk2D();
protected:
    vector<vector<vector<float> > > __create_batch(vector<float>& v);
    vector<float> __error();
    vector<float> __test_random();

};

#endif // GENERATE_SAMPLE_H
