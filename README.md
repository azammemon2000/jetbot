# Jetbot: Corrected Yahboom Hardware Driver & Autonomous Drone Docking

This repository provides the hardware-accurate `jetbot` Python package for the Yahboom Jetbot (`YB-EJF01` carrier board on NVIDIA Jetson Nano), along with the complete autonomous visual docking and wireless charging ingress pipeline.

---

## 1. Hardware Architecture & Pin Mapping

Standard Jetbot libraries assume an Adafruit TB6612 motor shield at address `0x60`, which produces channel crosstalk and fails to control Yahboom actuators. The Yahboom `YB-EJF01` carrier board utilizes an onboard PCA9685 PWM generator at address `0x40` driving three dedicated AM2857 H-bridge chips.

### I2C Bus & Addresses
* I2C Bus: `/dev/i2c-1` (Bus 1)
* PCA9685PW PWM Controller: Address `0x40` (64 decimal, address pins A0 to A5 tied to GND)
* Battery Telemetry ADC: Address `0x1b`

### Actuator Channel Assignments
* **Left Wheel Motor (J1)**:
  * Forward PWM: PCA9685 Channel 8 (`IN1A`)
  * Reverse PWM: PCA9685 Channel 9 (`IN1B`)
  * Power: 12V Battery Rail (`VM`)
* **Right Wheel Motor (J2)**:
  * Forward PWM: PCA9685 Channel 11 (`IN2B`)
  * Reverse PWM: PCA9685 Channel 10 (`IN2A`)
  * Power: 12V Battery Rail (`VM`)
* **Vertical Platform Scissor Lift (J3)**:
  * Up PWM: PCA9685 Channel 12 (`IN3A`)
  * Down PWM: PCA9685 Channel 13 (`IN3B`)
  * Power: Regulated 5V Rail (Operates independently of 12V battery switch)
* **Limit Switch Header (J4)**:
  * Upper Limit Switch: Jetson 40-pin Header Pin 11 (BCM GPIO 17 / Tegra GPIO50)
  * Lower Limit Switch: Jetson 40-pin Header Pin 12 (BCM GPIO 18 / Tegra GPIO79)
  * Pull-up: 3.3V via 2k resistors (Reads `0` when triggered, `1` when open)

---

## 2. Installation

Clone and install directly into your Python virtual environment (e.g. Python 3.11 or Python 3.6):

```bash
git clone https://github.com/azammemon2000/jetbot.git
cd jetbot
pip install -e .
```

Prerequisites:
```bash
pip install -r requirements.txt
```

---

## 3. Basic Usage

### Initializing the Robot
```python
from jetbot import Robot

robot = Robot()
```

### Wheel Motion
Both drive wheels are calibrated for straight forward translation:
```python
# Move forward for 1 second at 60% speed
robot.forward(0.6, duration=1.0)

# Continuous forward
robot.forward(0.5)

# Stop drive motors
robot.stop()

# Turn left on the spot
robot.left(0.5, duration=0.8)

# Turn right on the spot
robot.right(0.5, duration=0.8)

# Move backward
robot.backward(0.5, duration=1.0)
```

### Vertical Platform Up / Down Control
Controls the scissor lift motor on connector J3:
```python
# Raise platform up for 2 seconds
robot.up(duration=2.0)

# Lower platform down for 2 seconds
robot.down(duration=2.0)

# Emergency release / stop
robot.vertical_motors_stop()
```

---

## 4. Autonomous Drone Docking System

The script `autonomous_docking.py` provides a five-stage autonomous navigation and visual servoing pipeline to dock the UGV underneath a drone landing gear for inductive charging.

### Mission Pipeline
1. **Panoramic Target Acquisition**: 360 degree sweep to detect the drone landing gear from 5+ feet away.
2. **Visual Approach**: Closed-loop proportional tracking to approach within 1.5 feet of the structure.
3. **Barrier vs Gateway Classification**: Inspects the ground plane to detect whether the UGV is facing the closed horizontal skid barrier or the open ingress gateway.
4. **Orbital Flank Maneuver**: If facing the barrier, performs an orbital arc around the structure until aligned with the open entrance.
5. **Tunnel Ingress & Coil Centering**: Navigates dead-center through the landing gear archway and halts concentric with the primary charging coil.

### Running the Docking Mission
```bash
# Perception check without motor actuation
python autonomous_docking.py --dry-run

# Execute full autonomous docking run
python autonomous_docking.py
```

---

## 5. Utility Scripts
* `read_battery.py`: Reads instantaneous battery voltage from I2C ADC `0x1b`.
* `raise_platform.py`: Safely elevates platform to top position with limit switch detection.
* `jetbot_motors.py`: Zero-dependency standalone driver on PCA9685 `0x40`.
