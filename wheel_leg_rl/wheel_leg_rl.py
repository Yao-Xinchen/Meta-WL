import numpy as np
import os
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

class WheelLegRL(Node):
    def __init__(self):
        super().__init__('wheel_leg_rl')
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

        self._motor_pos = {"L_WHL": 0, "R_WHL": 0, "L_LEG": 0, "R_LEG": 0}
        self._motor_vel = {"L_WHL": 0, "R_WHL": 0, "L_LEG": 0, "R_LEG": 0}
        self._convert = MotorConverter()

        # scales
        self._ang_vel_scale = 0.25
        self._lin_vel_scale = 2.0
        self._height_scale = 5.0
        self._dof_vel_scale = 0.05
        self._dof_pos_scale = 1.0
        self._action_leg_scale = 0.1
        self._action_wheel_scale = 7.0

        # timeouts
        self._timeout_threshold = 0.2
        self._last_command_moment = rclpy.time.Time().nanoseconds
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
        leg_pos, leg_vel = self._convert.motor_to_leg(
            np.array([self._motor_pos["L_LEG"], self._motor_pos["R_LEG"]]),
            np.array([self._motor_vel["L_LEG"], self._motor_vel["R_LEG"]]))
        wheel_vel = self._convert.motor_to_wheel(np.array([self._motor_vel["L_WHL"], self._motor_vel["R_WHL"]]))
        # input to actor
        self._actor.input_dof_pos(np.array([leg_pos[0], leg_pos[1]]) * self._dof_pos_scale)
        self._actor.input_dof_vel(np.array([leg_vel[0], leg_vel[1], wheel_vel[0], wheel_vel[1]]) * self._dof_vel_scale)

    def _command_callback(self, msg: Move):
        self._actor.input_commands(np.array(
            [msg.vel_x * self._lin_vel_scale,
            msg.omega * self._ang_vel_scale,
            msg.height * self._height_scale]))
        self._last_command_moment = rclpy.time.Time().nanoseconds

    def _imu_callback(self, msg: Imu):
        # angular velocity
        x = msg.angular_velocity.x
        y = msg.angular_velocity.y
        z = msg.angular_velocity.z
        self._actor.input_base_ang_vel(np.array([x, y, z]) * self._ang_vel_scale)

        # gravity vector
        world_gravity = np.array([0, 0, -9.81])
        world_to_imu = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        imu_to_base = [0.99552905, 0.0, -0.09445582, 0.0]  # base's x is 10.84 degrees upper than imu's
        world_to_base = quat_multiply(imu_to_base, world_to_imu)
        base_gravity = quat_rotate_inverse(world_to_base, world_gravity)
        self._actor.input_projected_gravity(base_gravity)

    def _pub_callback(self):
        # get scaled action
        action = self._actor.output_action()
        leg_pos = np.array([action[0], action[1]]) * self._action_leg_scale
        wheel_vel = np.array([action[2], action[3]]) * self._action_wheel_scale
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
        if rclpy.time.Time().nanoseconds - self._last_command_moment > self._timeout_threshold * 1e9:
            # stop robot if no command received for a while
            self._actor.input_commands(np.array([0, 0, 0.2]))


def main(args=None):
    rclpy.init(args=args)
    wheel_leg_rl = WheelLegRL()
    rclpy.spin(wheel_leg_rl)
    rclpy.shutdown()


if __name__ == '__main__':
    main()