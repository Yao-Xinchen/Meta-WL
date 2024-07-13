import numpy as np
import rclpy
import rclpy.logging

class Converter:
    def __init__(self):
        self._deceleration = 20
        self._leg_offset = 0.245
        self._motor_offset = 0.0
        self._leg_max = 0.035
        self._leg_min = -0.125
    
    def leg_to_motor(self, leg_action_pos):
        # pos
        leg_action_pos = self._clip_leg(leg_action_pos)
        leg_real_pos = leg_action_pos + self._leg_offset
        motor_real_pos = self._leg_to_motor_real(leg_real_pos)
        motor_simu_pos = (motor_real_pos - self._motor_offset) * np.array([-1, 1])
        return motor_simu_pos
    
    def motor_to_leg(self, motor_fb_pos, motor_fb_vel):
        # pos
        motor_real_pos = motor_fb_pos * np.array([-1, 1]) + self._motor_offset
        leg_real_pos = self._motor_to_leg_real(motor_real_pos)
        leg_simu_pos = leg_real_pos - self._leg_offset
        # vel
        temp_dt = 0.001
        motor_real_pos_next = (motor_fb_pos + motor_fb_vel * temp_dt) * np.array([-1, 1]) + self._motor_offset
        leg_real_pos_next = self._motor_to_leg_real(motor_real_pos_next)
        leg_simu_vel = (leg_real_pos_next - leg_real_pos) / temp_dt  # = leg_real_vel
        return leg_simu_pos, leg_simu_vel
    
    def wheel_to_motor(self, wheel_vel):
        return np.array([wheel_vel[0] * -self._deceleration, wheel_vel[1] * self._deceleration])
    
    def motor_to_wheel(self, motor_vel):
        return np.array([motor_vel[0] / -self._deceleration, motor_vel[1] / self._deceleration])
    
    def _leg_to_motor_real(self, leg_real):
        # fit curve
        length = leg_real + self._leg_offset
        return -37.77 * length ** 3 + 26.70 * length ** 2 - 1.47 * length + 0.77 - self._motor_offset
    
    def _motor_to_leg_real(self, motor_real):
        # fit curve
        angle = motor_real + self._motor_offset
        return 0.1102 * angle ** 3 - 0.4610 * angle ** 2 + 0.848 * angle + -0.3540 - self._leg_offset
    
    def _clip_leg(self, leg):
        return np.clip(leg, self._leg_min, self._leg_max)
    