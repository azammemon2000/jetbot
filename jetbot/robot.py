import time
import traitlets
from traitlets.config.configurable import SingletonConfigurable
from Adafruit_MotorHAT.Adafruit_PWM_Servo_Driver import PWM
from .motor import AM2857Motor, Motor

class Robot(SingletonConfigurable):
    """
    Native Yahboom Jetbot Robot Class (YB-EJF01 carrier board).
    Hardware Mapping:
      PCA9685 I2C Address: 0x40 on Bus 1
      Left Wheel Motor (J1):  INA = Channel 9,  INB = Channel 8  (U1 AM2857)
      Right Wheel Motor (J2): INA = Channel 10, INB = Channel 11 (U2 AM2857)
      Vertical Lift (J3):     INA = Channel 12, INB = Channel 13 (U3 AM2857)
    """
    left_motor = traitlets.Instance(AM2857Motor)
    right_motor = traitlets.Instance(AM2857Motor)

    i2c_bus = traitlets.Integer(default_value=1).tag(config=True)
    i2c_addr = traitlets.Integer(default_value=0x40).tag(config=True)

    left_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)
    right_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)

    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        self.pwm = PWM(self.i2c_addr, i2c_bus=self.i2c_bus)
        self.pwm.setPWMFreq(1600)

        # Left motor: Channel 11 (INA) and 10 (INB)
        self.left_motor = AM2857Motor(self.pwm, pin_a=11, pin_b=10, alpha=self.left_motor_alpha)
        # Right motor: Channel 8 (INA) and 9 (INB)
        self.right_motor = AM2857Motor(self.pwm, pin_a=8, pin_b=9, alpha=self.right_motor_alpha)

        # Clear vertical lift motor channels (12 and 13)
        self.vertical_motors_stop()

    def set_motors(self, left_speed, right_speed):
        self.left_motor.value = left_speed
        self.right_motor.value = right_speed

    def forward(self, speed=1.0, duration=None):
        self.left_motor.value = speed
        self.right_motor.value = speed
        if duration is not None:
            time.sleep(float(duration))
            self.stop()

    def backward(self, speed=1.0, duration=None):
        self.left_motor.value = -speed
        self.right_motor.value = -speed
        if duration is not None:
            time.sleep(float(duration))
            self.stop()

    def left(self, speed=1.0, duration=None):
        self.left_motor.value = -speed
        self.right_motor.value = speed
        if duration is not None:
            time.sleep(float(duration))
            self.stop()

    def right(self, speed=1.0, duration=None):
        self.left_motor.value = speed
        self.right_motor.value = -speed
        if duration is not None:
            time.sleep(float(duration))
            self.stop()

    def stop(self):
        self.left_motor.value = 0.0
        self.right_motor.value = 0.0

    def close(self):
        self.stop()

    def up(self, duration=1.0, speed=1.0):
        """
        Elevate vertical lift platform UP on J3 (Channels 12 & 13).
        Accepts duration in seconds (e.g. robot.up(5) runs for 5s then stops).
        """
        pwm_val = int(min(max(abs(speed), 0.0), 1.0) * 4095)
        # UP: Channel 12 PWM, Channel 13 LOW
        self.pwm.setPWM(12, 0, pwm_val)
        self.pwm.setPWM(13, 0, 0)
        if duration is not None and float(duration) > 0:
            time.sleep(float(duration))
            self.vertical_motors_stop()

    def down(self, duration=1.0, speed=1.0):
        """
        Lower vertical lift platform DOWN on J3 (Channels 12 & 13).
        Accepts duration in seconds (e.g. robot.down(5) runs for 5s then stops).
        """
        pwm_val = int(min(max(abs(speed), 0.0), 1.0) * 4095)
        # DOWN: Channel 12 LOW, Channel 13 PWM
        self.pwm.setPWM(12, 0, 0)
        self.pwm.setPWM(13, 0, pwm_val)
        if duration is not None and float(duration) > 0:
            time.sleep(float(duration))
            self.vertical_motors_stop()

    def vertical_motors_stop(self):
        """
        Stop and release vertical lift motor on J3.
        """
        self.pwm.setPWM(12, 0, 0)
        self.pwm.setPWM(13, 0, 0)
