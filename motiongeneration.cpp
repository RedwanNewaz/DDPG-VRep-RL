#include "motiongeneration.h"

motionGeneration::motionGeneration(string weight)
{
    cout<<"loading drl weight "<< weight<<endl;
    drl = make_shared<KerasModel>(weight.c_str(),false);

}

vec motionGeneration::drlEnv(vec start, vec dest)
{
    /* keras deep learning model is converted to cpp model.
     The model weight is restored by json param file (look at the header for implementation)
     first we need to update the robot and goal state.
     The error vectors are computed as follow $e_p, e_q, e_p_dot, e_q_dot$
     where _p stands for position, _q for orientation and _dot for derivation.
     the output is a vector represnts 4 control signals
     */
    generate_sample g;
    g.update_robot(start);
    g.update_target(dest);
    auto sample = g.get_sample();
    return drl->compute_output(sample);
}

float motionGeneration::genTraj(int objectHandle, vec start, vec dest, float speed)
{
    /* create the reference for a quadrotor trajectory.
     Later, when is required we can acess to corresponding class.
     Implementation focus on multiple robots.
     */
    auto it = quadRef.find(objectHandle);

    if (it != quadRef.end()){
        quadRef.erase (it);
    }
    _updateStatus(true,objectHandle);

    
    unique_ptr<TrajRunnable> qTraj(new TrajRunnable(start,dest,speed));
    float cost = qTraj->cost();
    quadRef.insert(std::make_pair(objectHandle, std::move(qTraj)) );
    return cost;
}

bool motionGeneration::executeTraj(int objectHandle, vec& nextLocation, float dt)
{
    /* retrive quadRefrernce then update the nextLocation and
     finally return the trajectory completion status
     */
    return (__status[objectHandle]?quadRef[objectHandle]->executeTraj(nextLocation, dt):false);
    
}

void motionGeneration::_updateStatus(bool status,int objectHandle)
{
    auto sit = __status.find(objectHandle);
    if (sit != __status.end()){
        __status.erase(sit);
    }
    __status.insert(make_pair(objectHandle,status));
}
