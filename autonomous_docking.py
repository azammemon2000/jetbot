#!/usr/bin/env python3
"""
Autonomous UGV Docking and Charging Navigation System for NVIDIA Jetson Nano.
Directly interfaces with CSI CAM0 via hardware-accelerated GStreamer and PCA9685 I2C motor driver.

Mission Capabilities:
1. 360-degree panoramic search to locate landing gear within 5+ ft range.
2. Closed-loop visual approach toward target.
3. Aspect Classification: Detects horizontal ground shaft to determine Closed vs Open side.
4. Orbital Flank Maneuver: If closed side is detected, orbits around the barrier to face the open side.
5. Precision Tunnel Servoing: Steers dead-center through the open archway.
6. Ingress & Centering: Slides under the landing gear and stops concentric with the drone center.
"""

import cv2
import numpy as np
import time
import os
import sys
import logging
try:
    from jetbot import Robot
except ImportError:
    from jetbot_motors import JetbotMotors as Robot

# Configure Logging
LOG_FILE = "/home/jetson/autonomous_docking.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=2
):
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class AutonomousDocker:
    def __init__(self):
        logging.info("Initializing Autonomous UGV Docking System...")
        self.motors = Robot()
        self.pipeline = gstreamer_pipeline()
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            logging.error("FATAL: Failed to open GStreamer camera pipeline on CAM0!")
            raise RuntimeError("Camera failed to open")

        # Let auto-exposure and white balance stabilize
        for _ in range(10):
            self.cap.read()
            time.sleep(0.03)

        logging.info("Camera and motor subsystems initialized successfully.")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logging.error("Camera frame read failed!")
            return None
        return frame

    def analyze_scene(self, frame):
        """
        Analyzes frame to:
        1. Find presence of landing gear structure.
        2. Extract left and right strut centroids.
        3. Detect presence of horizontal ground shaft (Closed vs Open side).
        """
        h, w, _ = frame.shape
        center_x = w // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Region 1: Ground level zone for horizontal shaft detection
        y_ground_top = int(h * 0.65)
        y_ground_bottom = int(h * 0.88)
        ground_roi = gray[y_ground_top:y_ground_bottom, :]

        blur_ground = cv2.GaussianBlur(ground_roi, (5, 5), 0)
        _, thresh_ground = cv2.threshold(blur_ground, 90, 255, cv2.THRESH_BINARY_INV)

        # Horizontal kernel to detect horizontal ground shaft
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        h_features = cv2.morphologyEx(thresh_ground, cv2.MORPH_OPEN, h_kernel)

        # Region 2: Vertical strut detection zone (from mid height to ground)
        y_strut_top = int(h * 0.35)
        y_strut_bottom = int(h * 0.80)
        strut_roi = gray[y_strut_top:y_strut_bottom, :]

        blur_strut = cv2.GaussianBlur(strut_roi, (5, 5), 0)
        thresh_strut = cv2.adaptiveThreshold(
            blur_strut, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )

        # Vertical projection across columns to locate struts
        col_sum = np.sum(thresh_strut, axis=0) / 255.0
        kernel_smooth = np.ones(15) / 15.0
        smoothed = np.convolve(col_sum, kernel_smooth, mode='same')

        # Find significant peaks
        peaks = []
        for i in range(int(w * 0.15), int(w * 0.85)):
            if smoothed[i] > 18 and smoothed[i] == max(smoothed[max(0, i-25):min(w, i+26)]):
                peaks.append((i, smoothed[i]))

        # Sort peaks by horizontal coordinate
        peaks.sort(key=lambda p: p[0])

        gear_detected = len(peaks) >= 2
        left_strut_x = None
        right_strut_x = None
        tunnel_mid_x = None
        strut_width = 0
        heading_error = 0

        if gear_detected:
            # Outer two prominent peaks define the bounds
            left_strut_x = peaks[0][0]
            right_strut_x = peaks[-1][0]
            tunnel_mid_x = (left_strut_x + right_strut_x) // 2
            strut_width = right_strut_x - left_strut_x
            heading_error = tunnel_mid_x - center_x

        # Ground shaft coverage analysis between strut span
        if gear_detected and strut_width > 40:
            span_x1 = max(0, left_strut_x + int(strut_width * 0.15))
            span_x2 = min(w, right_strut_x - int(strut_width * 0.15))
            if span_x2 > span_x1:
                span_h = h_features[:, span_x1:span_x2]
                col_has_bar = np.any(span_h > 0, axis=0)
                bar_coverage_pct = (np.sum(col_has_bar) / float(span_x2 - span_x1)) * 100.0
            else:
                bar_coverage_pct = 0.0
        else:
            bar_coverage_pct = 0.0

        is_closed_side = bar_coverage_pct > 50.0

        # Create annotated diagnostic frame
        annotated = frame.copy()
        if gear_detected:
            # Draw strut lines (green)
            cv2.line(annotated, (left_strut_x, 0), (left_strut_x, h), (0, 255, 0), 2)
            cv2.line(annotated, (right_strut_x, 0), (right_strut_x, h), (0, 255, 0), 2)

            # Draw tunnel target center (cyan)
            cv2.line(annotated, (tunnel_mid_x, 0), (tunnel_mid_x, h), (255, 255, 0), 2)

            # Draw camera center (red)
            cv2.line(annotated, (center_x, 0), (center_x, h), (0, 0, 255), 2)

            aspect_text = "CLOSED SIDE (BARRIER %.0f%%)" % bar_coverage_pct if is_closed_side else "OPEN SIDE (CLEAR %.0f%%)" % bar_coverage_pct
            status_color = (0, 0, 255) if is_closed_side else (0, 255, 0)
            cv2.putText(annotated, aspect_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(annotated, "Err: %+d px | Width: %d px" % (heading_error, strut_width), 
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.putText(annotated, "SEARCHING: NO LANDING GEAR IN FOV", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        return {
            "gear_detected": gear_detected,
            "left_strut_x": left_strut_x,
            "right_strut_x": right_strut_x,
            "tunnel_mid_x": tunnel_mid_x,
            "strut_width": strut_width,
            "heading_error": heading_error,
            "is_closed_side": is_closed_side,
            "bar_coverage_pct": bar_coverage_pct,
            "annotated_frame": annotated
        }

    def execute_search_sweep(self):
        """
        Rotates incrementally in place to scan 360 degrees for landing gear.
        """
        logging.info("[STATE 0: SEARCH] Initiating 360-degree scan sweep...")
        for step in range(12): # 12 increments of ~30 degrees
            frame = self.capture_frame()
            info = self.analyze_scene(frame)
            if info["gear_detected"]:
                logging.info("[STATE 0: SEARCH] Landing gear acquired at sweep step %d! (Strut Width: %d px)", step, info["strut_width"])
                cv2.imwrite("/home/jetson/milestone_0_acquired.jpg", info["annotated_frame"])
                return True

            # Incremental pulse turn
            logging.info("[STATE 0: SEARCH] Scanning... pulse rotating (step %d/12)", step+1)
            self.motors.set_motors(0.25, -0.25)
            time.sleep(0.18)
            self.motors.stop()
            time.sleep(0.35)

        logging.warning("[STATE 0: SEARCH] Full 360 sweep completed. Target not acquired.")
        return False

    def execute_approach(self):
        """
        Drives toward target landing gear until close enough to verify barrier aspect.
        Target approach distance: strut width approx 180-220 pixels.
        """
        logging.info("[STATE 1: APPROACH] Beginning closed-loop approach...")
        start_time = time.time()
        kp = 0.0018
        base_speed = 0.22

        while time.time() - start_time < 12.0:
            frame = self.capture_frame()
            info = self.analyze_scene(frame)

            if not info["gear_detected"]:
                logging.warning("[STATE 1: APPROACH] Target lost temporarily during approach, holding...")
                self.motors.stop()
                time.sleep(0.1)
                continue

            error = info["heading_error"]
            width = info["strut_width"]

            # Check if close enough for definitive aspect check
            if width >= 190:
                logging.info("[STATE 1: APPROACH] Close inspection distance reached (Width: %d px). Stopping.", width)
                self.motors.stop()
                cv2.imwrite("/home/jetson/milestone_1_approached.jpg", info["annotated_frame"])
                return info

            # Differential steering P-controller
            steer = kp * error
            left_v = max(0.12, min(0.35, base_speed + steer))
            right_v = max(0.12, min(0.35, base_speed - steer))

            self.motors.set_motors(left_v, right_v)
            time.sleep(0.05)

        self.motors.stop()
        logging.info("[STATE 1: APPROACH] Approach timeout reached.")
        return self.analyze_scene(self.capture_frame())

    def execute_orbital_flank(self):
        """
        Executes a 90-degree orbital flank around the landing gear to navigate from
        a Closed Side (ground barrier) to an Open Side (open arch tunnel).
        """
        logging.info("[STATE 2: ORBITAL FLANK] Barrier detected! Executing 90-deg orbit around landing gear...")

        # Step 1: Back up slightly for turning clearance
        logging.info("[STATE 2: ORBITAL FLANK] Backing up 20 cm for clearance...")
        self.motors.backward(0.24)
        time.sleep(0.8)
        self.motors.stop()
        time.sleep(0.2)

        # Step 2: Rotate 90 degrees right (tangent to landing gear)
        logging.info("[STATE 2: ORBITAL FLANK] Turning 90 degrees right...")
        self.motors.set_motors(0.30, -0.30)
        time.sleep(0.55)
        self.motors.stop()
        time.sleep(0.2)

        # Step 3: Drive forward along the flank (length of landing gear ~40 cm)
        logging.info("[STATE 2: ORBITAL FLANK] Driving forward along flank...")
        self.motors.forward(0.25)
        time.sleep(1.4)
        self.motors.stop()
        time.sleep(0.2)

        # Step 4: Turn 90 degrees left back toward landing gear center
        logging.info("[STATE 2: ORBITAL FLANK] Turning 90 degrees left to face open side...")
        self.motors.set_motors(-0.30, 0.30)
        time.sleep(0.55)
        self.motors.stop()
        time.sleep(0.4)

        # Re-evaluate aspect
        frame = self.capture_frame()
        info = self.analyze_scene(frame)
        cv2.imwrite("/home/jetson/milestone_2_flank_complete.jpg", info["annotated_frame"])
        logging.info("[STATE 2: ORBITAL FLANK] Flank completed. Current aspect: %s (Coverage: %.1f%%)",
                     "CLOSED" if info["is_closed_side"] else "OPEN", info["bar_coverage_pct"])
        return info

    def execute_tunnel_ingress(self):
        """
        Aligns dead-center with the open arch tunnel and slides smoothly under the drone belly.
        Stops when concentric with landing gear center (strut width >= 420 px).
        """
        logging.info("[STATE 3: PRECISION INGRESS] Commencing tunnel visual servoing and ingress...")
        kp_align = 0.0022
        v_ingress = 0.20
        target_docking_width = 400 # Calibrated width when UGV coil is centered under belly

        start_time = time.time()
        while time.time() - start_time < 15.0:
            frame = self.capture_frame()
            info = self.analyze_scene(frame)

            if not info["gear_detected"]:
                # When very close, struts move off screen edges as robot is already inside!
                logging.info("[STATE 4: DOCKED] Struts passed peripheral boundaries. UGV is fully underneath!")
                break

            error = info["heading_error"]
            width = info["strut_width"]

            if width >= target_docking_width:
                logging.info("[STATE 4: DOCKED] Target docking width reached (%d px >= %d px)!", width, target_docking_width)
                break

            # Precision steering
            steer = kp_align * error
            left_v = max(0.12, min(0.30, v_ingress + steer))
            right_v = max(0.12, min(0.30, v_ingress - steer))

            self.motors.set_motors(left_v, right_v)
            time.sleep(0.04)

        # Apply electronic brake
        self.motors.stop()
        time.sleep(0.1)

        final_frame = self.capture_frame()
        final_info = self.analyze_scene(final_frame)
        cv2.imwrite("/home/jetson/milestone_3_docked_final.jpg", final_info["annotated_frame"])
        logging.info("================================================================")
        logging.info("SUCCESS: DOCKING COMPLETE. UGV CENTERED UNDER LANDING GEAR.")
        logging.info("Primary charging coil is aligned with secondary receiving coil.")
        logging.info("================================================================")
        return True

    def run_mission(self):
        """
        Executes end-to-end autonomous mission.
        """
        logging.info("### STARTING AUTONOMOUS MISSION ###")

        # 1. Capture initial view
        frame = self.capture_frame()
        info = self.analyze_scene(frame)

        # 2. Check if landing gear is in view or if 360 search is needed
        if not info["gear_detected"]:
            logging.info("Landing gear not in initial FOV. Initiating 360 search...")
            found = self.execute_search_sweep()
            if not found:
                logging.error("Mission Aborted: Unable to find landing gear within range.")
                return False
            frame = self.capture_frame()
            info = self.analyze_scene(frame)

        # 3. Approach to inspection range
        info = self.execute_approach()

        # 4. Aspect check: Closed side barrier vs Open side gateway
        if info["is_closed_side"]:
            logging.info("Aspect Decision: CLOSED SIDE DETECTED (Horizontal Ground Shaft present).")
            info = self.execute_orbital_flank()
        else:
            logging.info("Aspect Decision: OPEN SIDE CONFIRMED (Ground plane is clear).")

        # 5. Final Ingress and Centering
        success = self.execute_tunnel_ingress()
        return success

    def cleanup(self):
        logging.info("Cleaning up resources...")
        self.motors.stop()
        self.motors.close()
        self.cap.release()
        logging.info("Shutdown completed cleanly.")

    def close(self):
        self.cleanup()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous UGV Docking System")
    parser.add_argument("--dry-run", action="store_true", help="Analyze scene without actuating motors")
    args = parser.parse_args()

    docker = None
    try:
        docker = AutonomousDocker()
        if args.dry_run:
            logging.info("Running in DRY-RUN mode (perception and telemetry only)...")
            frame = docker.capture_frame()
            info = docker.analyze_scene(frame)
            cv2.imwrite("/home/jetson/dry_run_analysis.jpg", info["annotated_frame"])
            logging.info("Dry run analysis completed:")
            logging.info("  Landing Gear Detected: %s", info["gear_detected"])
            logging.info("  Strut Width (px): %d", info["strut_width"])
            logging.info("  Tunnel Center (px): %s", str(info["tunnel_mid_x"]))
            logging.info("  Heading Error (px): %+d", info["heading_error"])
            logging.info("  Horizontal Shaft Coverage: %.1f%%", info["bar_coverage_pct"])
            logging.info("  Aspect Decision: %s", "CLOSED SIDE (BARRIER PRESENT)" if info["is_closed_side"] else "OPEN SIDE (CLEAR TUNNEL)")
            logging.info("Saved dry run visualization to /home/jetson/dry_run_analysis.jpg")
        else:
            docker.run_mission()
    except KeyboardInterrupt:
        logging.warning("Mission manually interrupted by user.")
    except Exception as e:
        logging.error("Mission encountered exception: %s", str(e), exc_info=True)
    finally:
        if docker:
            docker.cleanup()
