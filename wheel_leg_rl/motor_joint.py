import numpy as np

class MotorJoint:
    def __init__(self):
        self._deceleration = 20
        self._leg_offset = 0.245  # dof_pos + leg_offset = leg_length
        self._motor_offset = 1.3  # dof_pos + motor_offset = joint_angle
    
    def leg_to_motor(self, leg_pos):
        # pos
        motor_pos = self._leg_to_motor_pos(leg_pos) * np.array([-1, 1])
        return motor_pos
    
    def motor_to_leg(self, motor_pos, motor_vel):
        # pos
        leg_pos = self._motor_to_leg_pos(motor_pos * np.array([-1, 1]))
        # vel
        temp_dt = 0.001
        leg_pos_next = self._motor_to_leg_pos((motor_pos + motor_vel * temp_dt) * np.array([-1, 1]))
        leg_vel = (leg_pos_next - leg_pos) / temp_dt
        return leg_pos, leg_vel
    
    def wheel_to_motor(self, wheel_vel):
        return np.array([wheel_vel[0] * -self._deceleration, wheel_vel[1] * self._deceleration])
    
    def motor_to_wheel(self, motor_vel):
        return np.array([motor_vel[0] / -self._deceleration, motor_vel[1] / self._deceleration])
    
    def _leg_to_motor_pos(self, leg):
        # fit curve
        length = leg + self._leg_offset
        return -37.77 * length ** 3 + 26.70 * length ** 2 - 1.47 * length + 0.77 - self._motor_offset
    
    def _motor_to_leg_pos(self, motor):
        # fit curve
        angle = motor + self._motor_offset
        return 0.1102 * angle ** 3 - 0.4610 * angle ** 2 - 0.848 * angle + -0.3540 - self._leg_offset
    