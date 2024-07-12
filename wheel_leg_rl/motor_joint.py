import numpy as np

class MotorJoint:
    def __init__(self):
        self._deceleration = 20
        self._leg_offset = 0.245
        self._motor_offset = 0.0 # TODO: find the correct value
    
    def leg_to_motor(self, leg_pos, leg_vel):
        # pos
        motor_pos = self._leg_to_motor_pos(leg_pos)
        # vel
        temp_dt = 0.001
        motor_pos_next = self._leg_to_motor_pos(leg_pos + leg_vel * temp_dt)
        motor_vel = (motor_pos_next - motor_pos) / temp_dt
        return motor_pos, motor_vel
    
    def motor_to_leg(self, motor_pos, motor_vel):
        # pos
        leg_pos = self._motor_to_leg_pos(motor_pos)
        # vel
        temp_dt = 0.001
        leg_pos_next = self._motor_to_leg_pos(motor_pos + motor_vel * temp_dt)
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
    