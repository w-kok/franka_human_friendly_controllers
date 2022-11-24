import os
import time
import threading
import mujoco_py
import quaternion
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped 
from mujoco_panda import PandaArm
from mujoco_panda.utils.tf import quatdiff_in_euler
# from mujoco_panda.utils.viewer_utils import render_frame

"""
Simple Cartesian Impedance control in MuJoCo from ROS
"""

# --------- Modify as required ------------
# Task-space controller parameters
# stiffness gains
P_pos = 1000.
P_ori = 30.
# damping gains
D_pos = 2.*np.sqrt(P_pos)
D_ori = 2.*np.sqrt(P_ori)
ctrl_rate=1000
render_rate = 100
# -----------------------------------------

class CartesianImpedanceController():
    def __init__(self):
        rospy.init_node('controller', anonymous=True)
        self.cart_pub = rospy.Publisher('/cartesian_pose', PoseStamped, queue_size=0)
        self.cart_sub= rospy.Subscriber('/equilibrium_pose', PoseStamped, self.equilibrium_callback)
    

        self.p = PandaArm.withTorqueActuators(render=True, compensate_gravity=True)

        print("The control rate is:", ctrl_rate)
        

        self.p.set_neutral_pose()
        self.p.step()
        time.sleep(0.1)
        curr_pos, curr_ori = self.p.ee_pose()
        self.goal_pos=curr_pos
        self.goal_ori=curr_ori
        ctrl_thread = threading.Thread(target=self.controller)
        ctrl_thread.start()
        self.render()


    def equilibrium_callback(self, data):
        self.goal_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.goal_ori = np.array([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])


    def controller(self):

        while True:
            now_c = time.time()
            curr_pos, curr_ori = self.p.ee_pose()
            curr_vel, curr_omg = self.p.ee_velocity()


            delta_pos = (self.goal_pos - curr_pos).reshape([3, 1])
            delta_ori = quatdiff_in_euler(curr_ori, self.goal_ori).reshape([3, 1])
            # print
            F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
                np.vstack([D_pos*(curr_vel).reshape([3, 1]),
                    D_ori*(curr_omg).reshape([3, 1])]) 

            tau = np.dot(self.p.jacobian().T, F).flatten().tolist() # the null space control is missing 

            self.p.set_joint_commands(tau, compensate_dynamics=True)

            self.p.step(render=False)

            #enforce the rate
            elapsed_c = time.time() - now_c
            sleep_time_c = (1./ctrl_rate) - elapsed_c
            if sleep_time_c > 0.0:
                time.sleep(sleep_time_c)

    def render(self):

        while True:
            now_r=time.time()
            #render_frame(self.p.viewer, robot_pos, robot_ori)
            #render_frame(p.viewer, target_pos, original_ori, alpha=0.2)

            self.p.render()

            ## Publish important variables to ros
            curr_pos, curr_ori = self.p.ee_pose()
            goal = PoseStamped()
            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "map"

            goal.pose.position.x = curr_pos[0]
            goal.pose.position.y = curr_pos[1]
            goal.pose.position.z = curr_pos[2]

            goal.pose.orientation.x = curr_ori[1]
            goal.pose.orientation.y = curr_ori[2]
            goal.pose.orientation.z = curr_ori[3]
            goal.pose.orientation.w = curr_ori[0]
            self.cart_pub.publish(goal)

            ## enforce the rate 
            elapsed_r = time.time() - now_r
            sleep_time_r = (1./render_rate) - elapsed_r

            if sleep_time_r> 0.0:
                time.sleep(sleep_time_r)

if __name__ == "__main__":
    run=CartesianImpedanceController()
