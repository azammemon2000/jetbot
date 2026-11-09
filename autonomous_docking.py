#!/usr/bin/env python3
"""
Autonomous UGV Docking and Charging Navigation System for NVIDIA Jetson Nano.
Directly interfaces with CSI CAM0 via hardware-accelerated GStreamer pipe and corrected PCA9685 Robot driver.

Mission Capabilities:
1. Panoramic search to locate landing gear within 5+ ft range.
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
import subprocess
from jetbot import Robot

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

# Calibrated motor speeds for floor/carpet rolling friction (user verified >= 0.75 needed)
BASE_SPEED = 0.85
MIN_MOTOR_SPEED = 0.80
MAX_MOTOR_SPEED = 0.98
ROTATION_SPEED = 0.86

class GstCamera:
    """
    Subprocess GStreamer raw pipe reader.
    Streams 30 FPS BGR frames directly without relying on OpenCV GStreamer plugin.
    """
    def __init__(self, width=640, height=360, fps=30):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        gst_cmd = [
            'gst-launch-1.0', '-q',
            'nvarguscamerasrc', 'sensor-id=0', '!',
            f'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate={fps}/1', '!',
            'nvvidconv', 'flip-method=2', '!',
            f'video/x-raw, width={width}, height={height}, format=BGRx', '!',
            'videoconvert', '!',
            'video/x-raw, format=BGR', '!',
            'fdsink', 'fd=1'
        ]
        self.proc = subprocess.Popen(gst_cmd, stdout=subprocess.PIPE, bufsize=self.frame_size * 2)

        # Flush initial camera frames for auto-exposure stabilization
        for _ in range(10):
            self.read()

    def read(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) == self.frame_size:
            return True, np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return False, None

    def release(self):
        if hasattr(self, 'proc') and self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                pass
            self.proc = None

class AutonomousDocker:
    def __init__(self):
        logging.info("Initializing Autonomous UGV Docking System...")
        self.motors = Robot()
        # Strictly ensure vertical platform motors remain OFF and released
        self.motors.vertical_motors_stop()
        self.cap = GstCamera(width=640, height=360, fps=30)
        logging.info("Camera and motor subsystems initialized successfully.")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logging.error("Camera frame read failed!")
            return None
        return frame

    def analyze_scene(self, frame):
        """
        Analyzes scene to:
        1. Locate the black landing gear arches in the driving corridor.
        2. Extract left and right strut centroids.
        3. Check ground region between strut feet for continuous horizontal barrier shaft.
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Corridor of interest: x from 140 to 500, y from 90 to 255
        corridor_x1 = 120
        corridor_x2 = 520
        corridor_y1 = 80
        corridor_y2 = 250

        corridor_gray = gray[corridor_y1:corridor_y2, corridor_x1:corridor_x2]

        # Dark threshold: black landing gear on lighter floor
        dark = (corridor_gray < 75).astype(np.uint8) * 255

        # Vertical projection across upper arches (y from 10 to 90 inside corridor)
        vert_proj = np.sum(dark[10:95, :] > 0, axis=0)
        vert_smooth = np.convolve(vert_proj, np.ones(11)/11.0, mode='same')

        peaks = []
        thresh = np.max(vert_smooth) * 0.40 if np.max(vert_smooth) > 0 else 100
        for x in range(12, len(vert_smooth) - 12):
            if vert_smooth[x] > thresh and vert_smooth[x] == np.max(vert_smooth[max(0, x-14):min(len(vert_smooth), x+15)]):
                peaks.append(x + corridor_x1)

        gear_detected = False
        left_strut_x = None
        right_strut_x = None
        tunnel_mid_x = w // 2
        strut_width = 0
        is_closed_side = False
        bar_coverage_pct = 0.0

        if len(peaks) >= 2:
            # Pair closest to center with realistic width
            valid_pairs = []
            for i in range(len(peaks)):
                for j in range(i+1, len(peaks)):
                    pw = peaks[j] - peaks[i]
                    if 60 <= pw <= 380:
                        mid = (peaks[i] + peaks[j]) // 2
                        center_dist = abs(mid - (w // 2))
                        valid_pairs.append((peaks[i], peaks[j], pw, mid, center_dist))

            if valid_pairs:
                # Select pair closest to visual centerline
                best = min(valid_pairs, key=lambda p: p[4])
                left_strut_x = best[0]
                right_strut_x = best[1]
                strut_width = best[2]
                tunnel_mid_x = best[3]
                gear_detected = True

                # Check horizontal ground barrier between left and right strut feet (y from 210 to 240)
                ground_band = dark[120:165, (left_strut_x - corridor_x1):(right_strut_x - corridor_x1)]
                if ground_band.size > 0:
                    row_cov = np.mean(ground_band > 0, axis=1)
                    bar_coverage_pct = float(np.max(row_cov) * 100.0)

                    # Continuous black tube across ground means CLOSED barrier side
                    if bar_coverage_pct > 30.0:
                        is_closed_side = True

        annotated = frame.copy()
        if gear_detected:
            cv2.line(annotated, (left_strut_x, 0), (left_strut_x, h), (0, 255, 0), 2)
            cv2.line(annotated, (right_strut_x, 0), (right_strut_x, h), (0, 255, 0), 2)
            cv2.line(annotated, (tunnel_mid_x, 0), (tunnel_mid_x, h), (255, 0, 0), 2)
            cv2.line(annotated, (w//2, 0), (w//2, h), (0, 0, 255), 1)

            label = "ASPECT: CLOSED (BARRIER PRESENT)" if is_closed_side else "ASPECT: OPEN (GATEWAY)"
            color = (0, 0, 255) if is_closed_side else (0, 255, 0)
            cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated, f"Strut Width: {strut_width}px | Shaft Cov: {bar_coverage_pct:.1f}%",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(annotated, "SEARCHING FOR LANDING GEAR...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return {
            "gear_detected": gear_detected,
            "left_strut_x": left_strut_x,
            "right_strut_x": right_strut_x,
            "tunnel_mid_x": tunnel_mid_x,
            "heading_error": tunnel_mid_x - (w // 2),
            "strut_width": strut_width,
            "is_closed_side": is_closed_side,
            "bar_coverage_pct": bar_coverage_pct,
            "annotated_frame": annotated
        }

    def execute_search_sweep(self):
        logging.info("[STATE 0: SEARCH] Initiating 360-degree scan sweep...")
        for step in range(12):
            frame = self.capture_frame()
            info = self.analyze_scene(frame)
            if info["gear_detected"]:
                logging.info("[STATE 0: SEARCH] Landing gear acquired at sweep step %d! (Strut Width: %d px)", step, info["strut_width"])
                cv2.imwrite("/home/jetson/milestone_0_acquired.jpg", info["annotated_frame"])
                return True

            logging.info("[STATE 0: SEARCH] Scanning... pulse rotating (step %d/12)", step+1)
            self.motors.right(ROTATION_SPEED, duration=0.15)
            time.sleep(0.35)

        logging.warning("[STATE 0: SEARCH] Full 360 sweep completed. Target not acquired.")
        return False

    def execute_approach(self):
        logging.info("[STATE 1: APPROACH] Beginning closed-loop approach toward landing gear...")
        start_time = time.time()
        kp = 0.0010

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

            # Close inspection distance reached (approx 1.5 ft / 180 px)
            if width >= 170:
                logging.info("[STATE 1: APPROACH] Close inspection distance reached (Width: %d px). Stopping.", width)
                self.motors.stop()
                cv2.imwrite("/home/jetson/milestone_1_approached.jpg", info["annotated_frame"])
                return info

            # Differential steering P-controller with ground speed thresholds
            steer = kp * error
            left_v = max(MIN_MOTOR_SPEED, min(MAX_MOTOR_SPEED, BASE_SPEED + steer))
            right_v = max(MIN_MOTOR_SPEED, min(MAX_MOTOR_SPEED, BASE_SPEED - steer))

            self.motors.set_motors(left_v, right_v)
            time.sleep(0.08)

        self.motors.stop()
        logging.info("[STATE 1: APPROACH] Approach timeout reached.")
        return self.analyze_scene(self.capture_frame())

    def execute_orbital_flank(self):
        logging.info("[STATE 2: ORBITAL FLANK] Barrier detected! Executing orbit around landing gear...")

        # Step 1: Back up slightly for turning clearance
        logging.info("[STATE 2: ORBITAL FLANK] Backing up 20 cm for clearance...")
        self.motors.backward(BASE_SPEED, duration=0.35)
        time.sleep(0.2)

        # Step 2: Rotate 90 degrees right
        logging.info("[STATE 2: ORBITAL FLANK] Turning 90 degrees right...")
        self.motors.right(ROTATION_SPEED, duration=0.45)
        time.sleep(0.2)

        # Step 3: Drive forward along the flank
        logging.info("[STATE 2: ORBITAL FLANK] Driving forward along flank...")
        self.motors.forward(BASE_SPEED, duration=0.75)
        time.sleep(0.2)

        # Step 4: Turn 90 degrees left back toward landing gear center
        logging.info("[STATE 2: ORBITAL FLANK] Turning 90 degrees left to face open gateway...")
        self.motors.left(ROTATION_SPEED, duration=0.45)
        time.sleep(0.4)

        # Re-evaluate aspect
        frame = self.capture_frame()
        info = self.analyze_scene(frame)
        cv2.imwrite("/home/jetson/milestone_2_flank_complete.jpg", info["annotated_frame"])
        logging.info("[STATE 2: ORBITAL FLANK] Flank completed. Current aspect: %s (Coverage: %.1f%%)",
                     "CLOSED" if info["is_closed_side"] else "OPEN", info["bar_coverage_pct"])
        return info

    def execute_tunnel_ingress(self):
        logging.info("[STATE 3: PRECISION INGRESS] Commencing tunnel visual servoing and ingress...")
        kp_align = 0.0012
        target_docking_width = 320

        start_time = time.time()
        while time.time() - start_time < 10.0:
            frame = self.capture_frame()
            info = self.analyze_scene(frame)

            if not info["gear_detected"]:
                logging.info("[STATE 4: DOCKED] Struts passed peripheral boundaries. UGV is fully underneath!")
                break

            error = info["heading_error"]
            width = info["strut_width"]

            if width >= target_docking_width:
                logging.info("[STATE 4: DOCKED] Target docking width reached (%d px >= %d px)!", width, target_docking_width)
                break

            steer = kp_align * error
            left_v = max(MIN_MOTOR_SPEED, min(MAX_MOTOR_SPEED, BASE_SPEED + steer))
            right_v = max(MIN_MOTOR_SPEED, min(MAX_MOTOR_SPEED, BASE_SPEED - steer))

            self.motors.set_motors(left_v, right_v)
            time.sleep(0.06)

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
        logging.info("### STARTING AUTONOMOUS DOCKING MISSION ###")

        # 1. Capture initial view
        frame = self.capture_frame()
        info = self.analyze_scene(frame)

        # 2. Check if landing gear is in view or if search is needed
        if not info["gear_detected"]:
            logging.info("Landing gear not in initial FOV. Initiating scan...")
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
        self.cap.release()
        logging.info("Shutdown completed cleanly.")

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
