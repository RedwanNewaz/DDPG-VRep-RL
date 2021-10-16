#include "config.h"
#include "plugin.h"
#include "simPlusPlus/Plugin.h"
#include "simPlusPlus/Handle.h"
#include "stubs.h"
#include "header.h"



unique_ptr<motionGeneration>motion;
bool __status=false;

void drlEnv(SScriptCallBack *p, char const* name, drlEnv_in *in_args, drlEnv_out *out_args)
{
    auto u_t = motion->drlEnv(in_args->start, in_args->dest);
    copy(u_t.begin(), u_t.end(), back_inserter(out_args->u_t));
}

void genTraj(SScriptCallBack *p, char const* name, genTraj_in *in_args, genTraj_out *out_args)
{
    cout<<"request for robot "<<in_args->objectHandle<<endl;
    float out = motion->genTraj(in_args->objectHandle, in_args->start, in_args->dest, in_args->speed);
    out_args->cost=out;
    __status=true;

}

void executeTraj(SScriptCallBack *p, char const* name, executeTraj_in *in_args, executeTraj_out *out_args)
{
    vec nextLocation{0,0,0};
    int out;
    if(__status)
        out = motion->executeTraj(in_args->objectHandle, nextLocation, in_args->dt);
    else
        out=0;
    out_args->status=out;
    copy(nextLocation.begin(), nextLocation.end(), back_inserter(out_args->pos));
}


class Plugin : public sim::Plugin
{
public:
    void onStart()
    {


        if(!registerScriptStuff())
            throw std::runtime_error("script stuff initialization failed");

        simSetModuleInfo(PLUGIN_NAME, 0, "DRL Quadrotor Motoin Pluggin", 0);
        simSetModuleInfo(PLUGIN_NAME, 1, __DATE__, 0);
    }
    void onSimulationAboutToStart(){
        cout<<"loading motion lib"<<endl;
        string weight = read_param();
        motion = std::make_unique<motionGeneration>(weight);

    }


};

SIM_PLUGIN(PLUGIN_NAME, PLUGIN_VERSION, Plugin)
//#include "stubsPlusPlus.cpp"
