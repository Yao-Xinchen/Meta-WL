import rclpy
from rclpy.node import Node

from device_interface.msg import MotorGoal
from device_interface.msg import MotorState
from behavior_interface.msg import Move
from geometry_msgs.msg import Vector3

class WheelLegRL(Node):
    def __init__(self) -> None:
        super().__init__('wheel_leg_rl')
        self._state_sub = self.create_subscription(MotorState, "motor_state", self._state_callback, 10)
        self._command_sub = self.create_subscription(Move, "move", self._command_callback, 10)
        self._euler_sub = self.create_subscription(Vector3, "euler_angles", self._euler_callback, 10)
        self._goal_pub = self.create_publisher(MotorGoal, "motor_goal", 10)
        self._pub_timer = self.create_timer(0.01, self._pub_callback)
        self.get_logger().info("WheelLegRL initialized.")

    def _state_callback(self, msg: MotorState) -> None:
        pass

    def _command_callback(self, msg: Move) -> None:
        pass

    def _euler_callback(self, msg: Vector3) -> None:
        pass

    def _pub_callback(self) -> None:
        pass


def main(args=None):
    rclpy.init(args=args)
    wheel_leg_rl = WheelLegRL()
    rclpy.spin(wheel_leg_rl)
    rclpy.shutdown()


if __name__ == '__main__':
    main()