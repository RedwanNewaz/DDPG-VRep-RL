#ifndef MOTIONGENERATION_H
#define MOTIONGENERATION_H
#include<iostream>
#include<string>
#include <map>

#include <iostream>
#include <fstream>
#include "ddpg_plugin/keras_model.h"
#include "ddpg_plugin/generate_sample.h"
#include "trajrunnable.h"



using namespace std;



class motionGeneration
{
public:
    motionGeneration(string drlWeight);
    vec drlEnv(vec start, vec dest);
    float genTraj(int objectHandle, vec start, vec dest , float speed);
    bool executeTraj(int objectHandle, vec& nextLocation, float dt);
private:
    shared_ptr<KerasModel>drl;
    map<int,unique_ptr<TrajRunnable>> quadRef;
    map<int,bool> __status;
protected:
    void _updateStatus(bool status, int objectHandle);
};



#endif // MOTIONGENERATION_H
