mapControlInput = function(u)
    --[[
        control input from ddpg with tanh activation
        u = [-1, -1]
        need to map [1224, 1280]
        when all properller u = 1225 quadrotor start moving up
    ]]
    local min_u, max_u, min_v, max_v
    min_u, max_u = -1, 1
    min_v, max_v = 1170, 1400
    local scale = (u - min_u) / (max_u - min_u)
    local v = scale * (max_v - min_v) + min_v
    return v
end


function get_state(objectHandle)
    -- generate state vector for input
    -- the state vector has following chronological order
    --e_p, e_q, e_p_dot, e_q_dot
    local p=sim.getObjectPosition(objectHandle,-1)
    local q=sim.getObjectOrientation(objectHandle,-1)
    v,w= sim.getObjectVelocity(objectHandle)

    local x_t={p[1],p[2],p[3],q[1],q[2],q[3],v[1],v[2],v[3],w[1],w[2],w[3]}
    return x_t
end

motorVelocities = simMotionQuadRL.drlEnv(current,goalPose)

