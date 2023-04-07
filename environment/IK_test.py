import numpy as np
import pybullet as p
import pybullet_data
import time

def IK_reacher():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
    goal_id = p.loadURDF('../assets/goal.urdf',[1,1,1])
    robot_id = p.loadURDF("xarm/xarm6_robot_white.urdf", basePosition=[0, 0, 0.63], useFixedBase=True,flags=p.URDF_USE_SELF_COLLISION)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

    p.setGravity(0,0,-9.81)

    # get properties
    num_joints = p.getNumJoints(robot_id) - 1
    joint_indices = range(1,7)
    ll = [-3.142,-3.14,-3.142,-3.142,-3.142,-3.142]
    #upper limits for null space (todo: set them to proper range)
    ul = [3.142,0,3.142,3.142,3.142,3.142,3.142]
    #joint ranges for null space (todo: set them to proper range)

    jointIds = []
    for i in joint_indices:
        jointInfo = p.getJointInfo(robot_id, i)
        jointName = jointInfo[1].decode("utf-8")
        if "finger" not in jointName:
            jointIds.append((i, jointName))

    _link_name_to_index = {p.getBodyInfo(robot_id)[0].decode('UTF-8'):-1,}
            
    for _id in range(p.getNumJoints(robot_id)):
        _name = p.getJointInfo(robot_id, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id

    print(jointIds)
    print(_link_name_to_index)

    while True:
        # Randomize goal
        x = np.random.uniform(0.2,0.5)
        y = np.random.uniform(-0.5,0.5)
        z = np.random.uniform(0.63,1)
        goal_pose = np.array([x,y,z])
        p.resetBasePositionAndOrientation(goal_id,goal_pose,[0,0,0,1])

        angle_noise_range = 0.02
        joint_states = p.getJointStates(robot_id, joint_indices)
        qpos = np.array([j[0] for j in joint_states])
        qpos += np.random.uniform(-angle_noise_range, angle_noise_range, 6)
        qvel = np.array([j[1] for j in joint_states])
        qvel += np.random.uniform(-angle_noise_range, angle_noise_range, 6)


        for i in range(100):
            time.sleep(0.01)

            joint_states = p.getJointStates(robot_id, joint_indices)
            joint_positions = np.array([j[0] for j in joint_states])
            desired_joint_positions = p.calculateInverseKinematics(robot_id, num_joints, p.getBasePositionAndOrientation(goal_id)[0],[1,0,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]

            error = desired_joint_positions - joint_positions
            for idx, pos in zip(joint_indices,desired_joint_positions):
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos,
                    #forces=torque
                )
            cont_table =p.getContactPoints(robot_id,table_id)
            cont_self = p.getContactPoints(robot_id,robot_id)
            
            # ee_pos = np.array(p.getLinkState(robot_id, 6)[0])
            # print(ee_pos, goal_pose)             
            #print(p.getBasePositionAndOrientation(goal)[0], p.getLinkState(robot_id,6)[0])
            #print(desired_joint_positions, joint_positions)
            p.stepSimulation()
            #print(qpos, qvel)

    home_pos = np.array([0.3, 0, 1])
    while True:
        p.resetBasePositionAndOrientation(goal_id,home_pos,[0,0,0,1])
        time.sleep(0.01)
        joint_states = p.getJointStates(robot_id, joint_indices)
        joint_positions = np.array([j[0] for j in joint_states])
        desired_joint_positions = p.calculateInverseKinematics(robot_id, num_joints, home_pos,[1,0,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
        
        error = desired_joint_positions - joint_positions
        for idx, pos in zip(joint_indices,desired_joint_positions):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos,
                #forces=torque
            )
        ee_pos = np.array(p.getLinkState(robot_id, 6)[0])
        #print(p.getBasePositionAndOrientation(goal)[0], p.getLinkState(robot_id,6)[0])
        #print(desired_joint_positions, joint_positions)
        p.stepSimulation()
        #print(qpos, qvel)

    # initial ee pos: (0.20700000000000002, -0.000640499408948619, 0.7415200047024912)


    

if __name__ == "__main__":
    IK_reacher()
