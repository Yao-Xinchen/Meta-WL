import rclpy
import numpy as np
from rclpy.node import Node
from wheel_leg_rl.wl_actor import WLActor

from device_interface.msg import MotorGoal
from device_interface.msg import MotorState
from behavior_interface.msg import Move
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu

class WheelLegRL(Node):
    def __init__(self) -> None:
        super().__init__('wheel_leg_rl')
        self._actor = WLActor("~/temp")
        self._state_sub = self.create_subscription(MotorState, "motor_state", self._state_callback, 10)
        self._command_sub = self.create_subscription(Move, "move", self._command_callback, 10)
        self._imu_sub = self.create_subscription(Imu, "imu", self._imu_callback, 10)
        self._euler_sub = self.create_subscription(Vector3, "euler_angles", self._euler_callback, 10)
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
        # TODO: check the sign of the angular velocity
        roll = msg.angular_velocity.x
        pitch = - msg.angular_velocity.y
        yaw = msg.angular_velocity.z
        self._actor.input_base_ang_vel([roll, pitch, yaw])

    def _euler_callback(self, msg: Vector3) -> None:
        # projected gravity
        gravity = 9.81
        roll, pitch, yaw = msg.x, msg.y, msg.z
        gravity_vector = np.array([0, 0, -gravity])
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x
        projected = R @ gravity_vector
        self._actor.input_projected_gravity(projected)

    def _pub_callback(self) -> None:
        action = self._actor.output_action()
        msg = MotorGoal()
        msg.motor_id[:4] = ["L_WHL", "R_WHL", "L_LEG", "R_LEG"]
        msg.goal_pos[:4] = [np.nan, np.nan, action[0], action[1]]
        msg.goal_vel[:4] = [action[2], action[3], np.nan, np.nan]
        msg.goal_tor[:4] = [np.nan, np.nan, np.nan, np.nan]
        self._goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    wheel_leg_rl = WheelLegRL()
    rclpy.spin(wheel_leg_rl)
    rclpy.shutdown()


if __name__ == '__main__':
    main()