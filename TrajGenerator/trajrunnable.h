#ifndef TRAJRUNNABLE_H
#define TRAJRUNNABLE_H
#include<iostream>
#include<vector>
#include<math.h>
#include<thread>
#include <numeric>
#include <string>
#include <functional>
#include "RapidTrajectoryGenerator.h"

using namespace RapidQuadrocopterTrajectoryGenerator;
using namespace std;
typedef vector<float> vec;
// Computes the distance between two std::vectors
template <typename T>
double    vectors_distance(const std::vector<T>& a, const std::vector<T>& b)
{
    vec    auxiliary;
    
    std::transform (a.begin(), a.end(), b.begin(), std::back_inserter(auxiliary),//
                    [](T element1, T element2) {return pow((element1-element2),2.0);});
    auxiliary.shrink_to_fit();
    
    return  sqrt(std::accumulate(auxiliary.begin(), auxiliary.end(), 0.0));
} // end template vectors_distance

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class TrajRunnable
{
public:
    TrajRunnable(vec,vec,float);
    ~TrajRunnable();
    
    bool executeTraj(vector<float>&pose, float dt);
    float cost();
    
private:
    bool __status;
    float T,t;
    Vec3 vel0,velf,acc0,accf;
//    RapidTrajectoryGenerator *trajectory;
    unique_ptr<RapidTrajectoryGenerator> trajectory;
    vec start,dest;
    thread cThread;
    
protected:
    void __timeToGo(float speed);
    void __computeTrajectory();
    bool __executeTraj(vector<float>&pose, float dt);
};

#endif // TRAJRUNNABLE_H
