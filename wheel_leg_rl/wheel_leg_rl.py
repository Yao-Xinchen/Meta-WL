import rclpy
import numpy as np
from rclpy.node import Node
from wheel_leg_rl.wl_actor import WLActor
from wheel_leg_rl.helpers import *

from device_interface.msg import MotorGoal
from device_interface.msg import MotorState
from behavior_interface.msg import Move
from sensor_msgs.msg import Imu
from array import array

class WheelLegRL(Node):
    def __init__(self) -> None:
        super().__init__('wheel_leg_rl')
        self._actor = WLActor("~/temp")
        self._state_sub = self.create_subscription(MotorState, "motor_state", self._state_callback, 10)
        self._command_sub = self.create_subscription(Move, "move", self._command_callback, 10)
        self._imu_sub = self.create_subscription(Imu, "imu", self._imu_callback, 10)
        self._goal_pub = self.create_publisher(MotorGoal, "motor_goal", 10)
        self._pub_timer = self.create_timer(0.01, self._pub_callback)

        self._actor.start()

        self._motor_pos = {}
        self._motor_vel = {}

        self.get_logger().info("WheelLegRL initialized.")

    def __del__(self) -> None:
        self._actor.stop()

    def _state_callback(self, msg: MotorState) -> None:
        for id, pos, vel in zip(msg.motor_id, msg.present_pos, msg.present_vel):
            self._motor_pos[id] = pos
            self._motor_vel[id] = vel
        
        self._actor.input_dof_pos([self._motor_pos["L_LEG"], self._motor_pos["R_LEG"]])
        self._actor.input_dof_vel([self._motor_vel["L_LEG"], self._motor_vel["R_LEG"], self._motor_vel["L_WHL"], self._motor_vel["R_WHL"]])

    def _command_callback(self, msg: Move) -> None:
        self._actor.input_commands([msg.vel_x, msg.vel_y, msg.omega])

    def _imu_callback(self, msg: Imu) -> None:
        x = msg.angular_velocity.x
        y = msg.angular_velocity.y
        z = msg.angular_velocity.z
        self._actor.input_base_ang_vel([x, y, z])

        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        gravity_vector = np.array([0, 0, -9.81])
        projected = quat_rotate_inverse(quat, gravity_vector)
        self._actor.input_projected_gravity(projected)

    def _pub_callback(self) -> None:
        action = self._actor.output_action()
        msg = MotorGoal()
        msg.motor_id[:4] = ["L_WHL", "R_WHL", "L_LEG", "R_LEG"]
        msg.goal_pos[:4] = array('d', [np.nan, np.nan, action[0], action[1]])
        msg.goal_vel[:4] = array('d', [action[2], action[3], np.nan, np.nan])
        msg.goal_tor[:4] = array('d', [np.nan, np.nan, np.nan, np.nan])
        self._goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    wheel_leg_rl = WheelLegRL()
    rclpy.spin(wheel_leg_rl)
    rclpy.shutdown()


if __name__ == '__main__':
    main()