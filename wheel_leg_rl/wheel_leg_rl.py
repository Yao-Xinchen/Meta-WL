import numpy as np
import os
import time
from filterpy.kalman import KalmanFilter
from array import array
from wheel_leg_rl.wl_actor import WLActor
from wheel_leg_rl.quaternion import *
from wheel_leg_rl.motor_converter import MotorConverter

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from device_interface.msg import MotorGoal
from device_interface.msg import MotorState
from behavior_interface.msg import Move
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

class WheelLegRL(Node):
    def __init__(self):
        super().__init__('wheel_leg_rl')
        time.sleep(3.0)
        package_path = get_package_share_directory('wheel_leg_rl')
        actor_path = os.path.join(package_path, 'model', 'actor.onnx')
        encoder_path = os.path.join(package_path, 'model', 'encoder.onnx')
        self._actor = WLActor(actor_path, encoder_path)

        # subscriptions and publications
        self._state_sub = self.create_subscription(MotorState, "motor_state", self._state_callback, 10)
        self._command_sub = self.create_subscription(Move, "move", self._command_callback, 10)
        self._imu_sub = self.create_subscription(Imu, "imu", self._imu_callback, 10)
        self._goal_pub = self.create_publisher(MotorGoal, "motor_goal", 10)
        self._pub_timer = self.create_timer(0.005, self._pub_callback)

        self._debug_pub = self.create_publisher(Float64MultiArray, "debug", 10)

        # buffers
        self._motor_pos = {"L_WHL": 0, "R_WHL": 0, "L_LEG": 0, "R_LEG": 0}
        self._motor_vel = {"L_WHL": 0, "R_WHL": 0, "L_LEG": 0, "R_LEG": 0}

        # converter
        self._convert = MotorConverter()

        # scales
        self._ang_vel_scale = 0.25
        self._lin_vel_scale = 2.0
        self._height_scale = 5.0
        self._dof_vel_scale = 0.05
        self._dof_pos_scale = 1.0
        self._action_leg_scale = 0.1
        self._action_wheel_scale = 7.0
        self._command_scale = np.array([self._lin_vel_scale, self._ang_vel_scale, self._height_scale])

        # filters
        # too much noise in motor velocity
        vel_filter = KalmanFilter(dim_x=4, dim_z=4)
        vel_filter.x = np.array([0.0, 0.0, 0.0, 0.0])
        vel_filter.F = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        vel_filter.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        vel_filter.P *= 1000
        vel_filter.R *= 100
        self._vel_filter = vel_filter

        # timeouts
        self._timeout_threshold = 0.2
        self._last_command_moment = self.get_clock().now()
        self._timeout_timer = self.create_timer(0.1, self._timeout_callback)

        self.get_logger().info("WheelLegRL initialized.")

    def __del__(self):
        self._actor.stop()

    def _state_callback(self, msg: MotorState):
        # record to buffer
        for id, pos, vel in zip(msg.motor_id, msg.present_pos, msg.present_vel):
            self._motor_pos[id] = pos
            self._motor_vel[id] = vel

        # convert to dof
        leg_motor_pos = np.array([self._motor_pos["L_LEG"], self._motor_pos["R_LEG"]])
        leg_motor_vel = np.array([self._motor_vel["L_LEG"], self._motor_vel["R_LEG"]])
        wheel_motor_vel = np.array([self._motor_vel["L_WHL"], self._motor_vel["R_WHL"]])
        leg_pos, leg_vel = self._convert.motor_to_leg(leg_motor_pos, leg_motor_vel)
        wheel_vel = self._convert.motor_to_wheel(wheel_motor_vel)
        # self._debug_pub.publish(Float64MultiArray(data=leg_motor_vel))

        # input to actor
        # pos
        self._actor.input_dof_pos(leg_pos * self._dof_pos_scale)
        # self._debug_pub.publish(Float64MultiArray(data=leg_pos))
        # vel
        self._vel_filter.predict()
        self._vel_filter.update(np.concatenate((leg_vel, wheel_vel)))
        self._actor.input_dof_vel(self._vel_filter.x * self._dof_vel_scale)
        # self._debug_pub.publish(Float64MultiArray(data=self._vel_filter.x))

    def _command_callback(self, msg: Move):
        command = np.array([msg.vel_x, msg.omega, msg.height])
        self._actor.input_commands(command * self._command_scale)
        self._last_command_moment = self.get_clock().now()

    def _imu_callback(self, msg: Imu):
        # gravity vector
        world_gravity = np.array([0, 0, -1.0])  # unit vector pointing down
        world_to_imu = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        imu_to_base = euler_to_quaternion(0, 10.0 * np.pi / 180, 0)
        imu_gravity = quat_rotate_inverse(world_to_imu, world_gravity)
        base_gravity = quat_rotate(imu_to_base, imu_gravity)
        self._actor.input_projected_gravity(base_gravity)
        # self._debug_pub.publish(Float64MultiArray(data=base_gravity))

        # angular velocity
        ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        ang_vel = quat_rotate_inverse(world_to_imu, ang_vel)
        self._actor.input_base_ang_vel(ang_vel * self._ang_vel_scale)
        # self._debug_pub.publish(Float64MultiArray(data=ang_vel))

    def _pub_callback(self):
        # get scaled action
        action = self._actor.output_action()
        leg_pos = np.array([action[0], action[1]]) * self._action_leg_scale
        wheel_vel = np.array([action[2], action[3]]) * self._action_wheel_scale
        # data = np.concatenate((leg_pos, wheel_vel))
        # self._debug_pub.publish(Float64MultiArray(data=data))
        # convert to motor
        leg_motor_pos = self._convert.leg_to_motor(leg_pos)
        wheel_motor_vel = self._convert.wheel_to_motor(wheel_vel)
        # publish
        msg = MotorGoal()
        msg.motor_id[:4] = ["L_WHL", "R_WHL", "L_LEG", "R_LEG"]
        msg.goal_pos[:4] = array('d', [np.nan, np.nan, leg_motor_pos[0], leg_motor_pos[1]])
        msg.goal_vel[:4] = array('d', [wheel_motor_vel[0], wheel_motor_vel[1], np.nan, np.nan])
        msg.goal_tor[:4] = array('d', [np.nan, np.nan, np.nan, np.nan])
        self._goal_pub.publish(msg)

    def _timeout_callback(self):
        # check timeout regularly
        if self.get_clock().now().nanoseconds - self._last_command_moment.nanoseconds > self._timeout_threshold * 1e9:
            # stop robot if no command received for a while
            self._actor.input_commands(np.array([0.0, 0.0, 0.12]) * self._command_scale)


def main(args=None):
    rclpy.init(args=args)
    wheel_leg_rl = WheelLegRL()
    rclpy.spin(wheel_leg_rl)
    rclpy.shutdown()


if __name__ == '__main__':
    main()