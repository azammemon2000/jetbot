import atexit
import traitlets
from traitlets.config.configurable import Configurable
from Adafruit_MotorHAT.Adafruit_PWM_Servo_Driver import PWM

class AM2857Motor(Configurable):
    """
    Motor driver for Yahboom AM2857 H-bridge driven by PCA9685.
    AM2857 uses two inputs:
      Forward:  INA = PWM, INB = 0
      Backward: INA = 0,   INB = PWM
      Release:  INA = 0,   INB = 0
      Brake:    INA = 4095,INB = 4095
    """
    value = traitlets.Float()
    alpha = traitlets.Float(default_value=1.0).tag(config=True)
    beta = traitlets.Float(default_value=0.0).tag(config=True)

    def __init__(self, pwm_driver, pin_a, pin_b, *args, **kwargs):
        super(AM2857Motor, self).__init__(*args, **kwargs)
        self._pwm = pwm_driver
        self._pin_a = pin_a
        self._pin_b = pin_b
        self._release()
        atexit.register(self._release)

    @traitlets.observe('value')
    def _observe_value(self, change):
        self._write_value(change['new'])

    def _write_value(self, value):
        mapped = self.alpha * value + self.beta
        speed = int(min(max(abs(mapped), 0.0), 1.0) * 4095)
        if mapped > 0.001:
            self._pwm.setPWM(self._pin_a, 0, speed)
            self._pwm.setPWM(self._pin_b, 0, 0)
        elif mapped < -0.001:
            self._pwm.setPWM(self._pin_a, 0, 0)
            self._pwm.setPWM(self._pin_b, 0, speed)
        else:
            self._release()

    def _release(self):
        self._pwm.setPWM(self._pin_a, 0, 0)
        self._pwm.setPWM(self._pin_b, 0, 0)

# Backward-compatibility alias
Motor = AM2857Motor
