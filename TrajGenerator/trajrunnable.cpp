#include "trajrunnable.h"

TrajRunnable::TrajRunnable(vec start, vec dest, float speed):start(start),dest(dest)
{
    vel0=velf=acc0=accf=Vec3(0, 0, 0);
    
    __timeToGo(speed);
    __computeTrajectory();
    
}

TrajRunnable::~TrajRunnable()
{
    
}



void TrajRunnable::__computeTrajectory()
{
    /* this one is heaving lifting function so we will
     * use a thread to compute trajectory. Later we only
     * use the member functions of that trajectory which
     * are already computed. To access the members first
     * we need to store the trajecory in this object
     */
    __status = true;
    t=0;
    double Tf = T;
    cout<<"generate trajectory"<<endl;
    
    Vec3 pos0 = Vec3(start[0], start[1], start[2]); //position
    Vec3 posf = Vec3(dest[0], dest[1], dest[2]); //position
    
    
    //Define how gravity lies in our coordinate system
    Vec3 gravity = Vec3(0,0,-9.81);//[m/s**2]
    RapidTrajectoryGenerator traj(pos0, vel0, acc0, gravity);
    traj.SetGoalPosition(posf);
    traj.SetGoalVelocity(velf);
    traj.SetGoalAcceleration(accf);
    
    //generate trajectory
//        traj.Generate(Tf);
    std::thread t(&RapidTrajectoryGenerator::Generate,ref(traj),Tf);
    t.join();
//    trajectory = new RapidTrajectoryGenerator(traj);
    trajectory = make_unique<RapidTrajectoryGenerator>(traj);
    
    
}

bool TrajRunnable::executeTraj(vector<float> &pose, float dt)
{
    /* the key idea is to sequentially access to the trajectory elements.
     * given the time index it can provide the corresponding location.
     * dt here can be obtained from simulation state.
     */
    return (__status?__executeTraj(pose,dt):false);
    
}

float TrajRunnable::cost()
{
    // cost of the total trajectory. Lower is better.
    return trajectory->GetCost();
    
}

void TrajRunnable::__timeToGo(float speed)
{
    auto pathLength= vectors_distance(start,dest);
    T= pathLength/speed;
    cout<<"time to go is "<<T<<endl;
}

bool TrajRunnable::__executeTraj(vector<float> &pose, float dt)
{
    t+=dt; // update time
    //    cout<<"update pose vector "<<t;
    //    cout << " Total cost = " << trajectory->GetCost() << "\n";
    __status= (t<=T); // update flag
    if(__status)
    {
        auto posi=trajectory->GetPosition(t);
        for(int i(0);i<3;i++)pose[i]=posi[i];
        
    }
    else
    {
//        delete trajectory;
        cout<<"execution completed. I'm destroying trajectory...."<<endl;
    }
    return __status;
}
