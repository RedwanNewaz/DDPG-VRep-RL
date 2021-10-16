getContorlPoint = function(pos)
    ptData={pos[1],pos[2],pos[3],0.0,0.0,0.0,1.0,0,0,1.0,1.0}
    return ptData

end



    pathHandle=sim.createPath(-1,intParams,nil,nil)
    -- add robot location first
    local ptData = getContorlPoint(rpos)
    result=sim.insertPathCtrlPoints(pathHandle,0,0,1,ptData)
    

    for i=1,#path,7 do
        pos={path[i],path[i+1],path[i+2]}
        ptData = getContorlPoint(pos)
        
        result=sim.insertPathCtrlPoints(pathHandle,0,0,1,ptData)
        table.insert(lines,pos)
        
    end
    
    for j =1, #lines-1, 1 do 
        A= lines[j]
        B = lines[j+1]
        C= {A[1],A[2],A[3],B[1],B[2],B[3]}
        --sim.addDrawingObjectItem(lineContainer,C)
    end

    f = lines[#lines]
    -- add goal location last
    goalPos = sim.getObjectPosition(goalHandle,-1)
    ptData = getContorlPoint(goalPos)
    result=sim.insertPathCtrlPoints(pathHandle,0,0,1,ptData)
  