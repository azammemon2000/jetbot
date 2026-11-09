#!/usr/bin/env python3
"""
Safe Platform Raiser for Yahboom Jetbot.
Uses hardware limit switch (BCM pin 17) to safely elevate the platform to the top position.
"""

import time
import RPi.GPIO as GPIO
from Adafruit_MotorHAT import Adafruit_MotorHAT

UP_LIMIT_PIN = 17
DOWN_LIMIT_PIN = 18

def main():
    print("Initializing GPIO limit switches...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(UP_LIMIT_PIN, GPIO.IN)
    GPIO.setup(DOWN_LIMIT_PIN, GPIO.IN)

    print("Upper limit state (pin 17):", GPIO.input(UP_LIMIT_PIN))
    print("Lower limit state (pin 18):", GPIO.input(DOWN_LIMIT_PIN))

    if GPIO.input(UP_LIMIT_PIN) == 0:
        print("Platform is already at the TOP limit. No movement needed.")
        return

    print("Initializing MotorHAT at 0x40 on i2c-1...")
    mh = Adafruit_MotorHAT(addr=0x40, i2c_bus=1)
    m3 = mh.getMotor(3)
    m4 = mh.getMotor(4)

    print("Elevating platform UP until upper limit switch triggers...")
    # Speed 200 out of 255
    m3.setSpeed(200)
    m4.setSpeed(200)
    m3.run(Adafruit_MotorHAT.FORWARD)
    m4.run(Adafruit_MotorHAT.FORWARD)

    start_time = time.time()
    try:
        while time.time() - start_time < 4.0:
            if GPIO.input(UP_LIMIT_PIN) == 0:
                print("Upper limit reached! Stopping motor.")
                break
            time.sleep(0.05)
    finally:
        m3.run(Adafruit_MotorHAT.RELEASE)
        m4.run(Adafruit_MotorHAT.RELEASE)
        GPIO.cleanup()
        print("Platform lift motor stopped and released.")

if __name__ == '__main__':
    main()
