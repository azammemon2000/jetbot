#!/usr/bin/env python3
"""
Verified Motor Driver for Yahboom Jetbot UGV.
Uses the tested Adafruit_MotorHAT driver configured at I2C address 0x40 on bus 1.
"""

import time
import atexit
from Adafruit_MotorHAT import Adafruit_MotorHAT

class JetbotMotors:
    def __init__(self, addr=0x40, i2c_bus=1):
        self.hat = Adafruit_MotorHAT(addr=addr, i2c_bus=i2c_bus)
        self.motor_left = self.hat.getMotor(1)  # Motor Port 1 (Left Wheel)
        self.motor_right = self.hat.getMotor(2) # Motor Port 2 (Right Wheel)
        # Lock and release auxiliary channels 3 and 4 so lift motor never engages
        for aux in (3, 4):
            try:
                self.hat.getMotor(aux).run(Adafruit_MotorHAT.RELEASE)
            except Exception:
                pass
        self.stop()
        atexit.register(self.stop)

    def set_motors(self, left_speed, right_speed):
        """
        Set speed for left and right motors: range -1.0 to 1.0.
        """
        left_speed = max(-1.0, min(1.0, float(left_speed)))
        right_speed = max(-1.0, min(1.0, float(right_speed)))

        left_pwm = int(abs(left_speed) * 255)
        right_pwm = int(abs(right_speed) * 255)

        self.motor_left.setSpeed(left_pwm)
        self.motor_right.setSpeed(right_pwm)

        if left_speed > 0:
            self.motor_left.run(Adafruit_MotorHAT.FORWARD)
        elif left_speed < 0:
            self.motor_left.run(Adafruit_MotorHAT.BACKWARD)
        else:
            self.motor_left.run(Adafruit_MotorHAT.RELEASE)

        if right_speed > 0:
            self.motor_right.run(Adafruit_MotorHAT.FORWARD)
        elif right_speed < 0:
            self.motor_right.run(Adafruit_MotorHAT.BACKWARD)
        else:
            self.motor_right.run(Adafruit_MotorHAT.RELEASE)

    def forward(self, speed=0.6):
        self.set_motors(speed, speed)

    def backward(self, speed=0.6):
        self.set_motors(-speed, -speed)

    def left(self, speed=0.5):
        self.set_motors(-speed, speed)

    def right(self, speed=0.5):
        self.set_motors(speed, -speed)

    def stop(self):
        if self.motor_left:
            self.motor_left.run(Adafruit_MotorHAT.RELEASE)
        if self.motor_right:
            self.motor_right.run(Adafruit_MotorHAT.RELEASE)

    def close(self):
        self.stop()

if __name__ == '__main__':
    print("Initializing JetbotMotors...")
    motors = JetbotMotors()
    print("Testing forward for 1 second...")
    motors.forward(0.7)
    time.sleep(1.0)
    print("Testing backward for 1 second...")
    motors.backward(0.7)
    time.sleep(1.0)
    motors.stop()
    print("Test complete.")
