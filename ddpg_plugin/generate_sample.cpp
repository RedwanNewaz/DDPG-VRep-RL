#include "generate_sample.h"
#include <algorithm>
generate_sample::generate_sample()
{
    __robot_state.reserve(STATE_SIZE);
    __target_state.reserve(STATE_SIZE);
    for (int i(0);i<STATE_SIZE;i++)
        __robot_state[i]=__target_state[i]=0.0;
}

std::shared_ptr<DataChunk> generate_sample::get_sample()
{
    /* Preditction only possible with DataChunk type variable.
     * we need smart pointer to automatically delete the pointer
     * DataChunk *sample = new DataChunk2D();
     * TODO: replace __test_random() to __error()
     */

    std::shared_ptr<DataChunk> sample(new DataChunk2D());
    auto e = __error();
    auto y_ret = __create_batch(e);
    sample->set_data(y_ret);
    return sample;
}

vector<vector<vector<float> > > generate_sample::__create_batch(vector<float> &v)
{
    /* INPUT = 1D vector with dimension (12,)
     * Here we convert 1D array to 3D tuple with dimension (1x1x12)
     * since we cannot use 1D array to obtain prediction.
     * OUTPUT = 3D tuple with dimension (1x1x12)
     */
    vector<vector<vector<float> > > y_ret;

    for(unsigned int i = 0; i < 1; ++i) {
      vector<vector<float> > tmp_y;
      for(unsigned int j = 0; j < 1; ++j) {
        tmp_y.push_back(v);
      }
      y_ret.push_back(tmp_y);
    }
    return y_ret;
}

vector<float> generate_sample::__error()
{
    vector<float> err(STATE_SIZE);
    for (int i(0);i<STATE_SIZE;i++){
        err[i]=__target_state[i]-__robot_state[i];
    }
    return err;
}

vector<float> generate_sample::__test_random()
{
    /* generate random variable for testing purpose
     */
    std::vector<float> v(STATE_SIZE);
    auto foo =[](){return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);};// lambda function for random number
    std::generate(v.begin(), v.end(), foo);
    return v;
}

void generate_sample::update_robot(vector<float>& x_t) { 
    for(int i=0;i<STATE_SIZE;i++)
        __robot_state[i]=x_t[i];
    
}

void generate_sample::update_target(vector<float> &x_t) {
    for(int i=0;i<STATE_SIZE;i++)
        __target_state[i]=x_t[i];
}





