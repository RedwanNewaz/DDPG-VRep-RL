
#include "config.h"
#include "plugin.h"
#include "simPlusPlus/Plugin.h"
#include "simPlusPlus/Handle.h"
#include "stubs.h"

// an example data structure to hold data across multiple calls
struct ExampleObject
{
    int a = 0;
    int b = 0;
    std::vector<int> seq;
};

// conversion from C pointer to string handle and vice-versa:
template<> std::string sim::Handle<ExampleObject>::tag() { return "Example.Object"; }

class Plugin : public sim::Plugin
{
public:
    void onStart()
    {
        if(!registerScriptStuff())
            throw std::runtime_error("script stuff initialization failed");

        setExtVersion("Example Plugin Skeleton");
        setBuildDate(BUILD_DATE);
    }

    void onScriptStateDestroyed(int scriptID)
    {
        for(auto obj : handles.find(scriptID))
            delete handles.remove(obj);
    }

    void createObject(createObject_in *in, createObject_out *out)
    {
        auto obj = new ExampleObject;

        out->handle = handles.add(obj, in->_.scriptID);
    }

    void destroyObject(destroyObject_in *in, destroyObject_out *out)
    {
        auto obj = handles.get(in->handle);

        delete handles.remove(obj);
    }

    void setData(setData_in *in, setData_out *out)
    {
        auto obj = handles.get(in->handle);

        if(!obj->seq.empty())
            sim::addLog(sim_verbosity_warnings, "current sequence not empty");

        obj->a = in->a;
        obj->b = in->b;
    }

    void compute(compute_in *in, compute_out *out)
    {
        auto obj = handles.get(in->handle);

        obj->seq.push_back(obj->a + obj->b);
        obj->a = obj->b;
        obj->b = obj->seq.back();
        out->currentSize = obj->seq.size();
    }

    void getOutput(getOutput_in *in, getOutput_out *out)
    {
        auto obj = handles.get(in->handle);

        out->output = obj->seq;
    }

private:
    sim::Handles<ExampleObject> handles;
};

SIM_PLUGIN(PLUGIN_NAME, PLUGIN_VERSION, Plugin)
#include "stubsPlusPlus.cpp"
