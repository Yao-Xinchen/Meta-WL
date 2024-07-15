import numpy as np
import rclpy

class MotorConverter:
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
        motor_real_pos = self._m(leg_real_pos)
        motor_simu_pos = (motor_real_pos - self._motor_offset) * np.array([-1, 1])
        return motor_simu_pos
    
    def motor_to_leg(self, motor_fb_pos, motor_fb_vel):
        # pos
        motor_real_pos = motor_fb_pos * np.array([-1, 1]) + self._motor_offset
        leg_real_pos = self._l(motor_real_pos)
        leg_simu_pos = leg_real_pos - self._leg_offset
        # vel
        motor_real_vel = motor_fb_vel * np.array([-1, 1])
        leg_real_vel = self._dl_dm(motor_real_pos) * motor_real_vel  # dl/dt = dl/dm * dm/dt
        leg_simu_vel = leg_real_vel
        return leg_simu_pos, leg_simu_vel
    
    def wheel_to_motor(self, wheel_action_vel):
        return np.array([wheel_action_vel[0] * -self._deceleration, wheel_action_vel[1] * self._deceleration])
    
    def motor_to_wheel(self, motor_fb_vel):
        return np.array([motor_fb_vel[0] / -self._deceleration, motor_fb_vel[1] / self._deceleration])

    def _m(self, l):
        # calculate motor position from leg position
        m = -37.77 * l ** 3 + 26.70 * l ** 2 - 1.47 * l + 0.77
        return m
    
    def _dm_dl(self, l):
        # derivative of m over l
        dm_dl = -37.77 * 3 * l ** 2 + 26.70 * 2 * l - 1.47
        return dm_dl
    
    def _l(self, m):
        # calculate leg position from motor
        l = 0.1102 * m ** 3 - 0.4610 * m ** 2 + 0.848 * m + -0.3540
        return l
    
    def _dl_dm(self, m):
        # derivative of l over m
        dl_dm = 0.1102 * 3 * m ** 2 - 0.4610 * 2 * m + 0.848
        return dl_dm
    
    def _clip_leg(self, leg):
        return np.clip(leg, self._leg_min, self._leg_max)
    