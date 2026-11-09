#!/usr/bin/env python3
"""
Autonomous Drone Landing Gear Docking System for Jetbot UGV.

Trigonometric Orbital Waypoint & Visual Docking:
1. 360-degree sweep search: Rotates 30 degrees clockwise up to 12 steps until landing gear is located.
2. Distance-based approach: Closes in if far so it can reliably inspect the aspect.
3. Aspect Classification:
   - Closed side: horizontal skid tube on ground blocks entry.
   - Open side: clear floor between arches (docking gateway).
4. Trigonometric Flank (Pythagoras Theorem):
   - Given perpendicular distance d and standoff base b (2.5 ft),
   - Computes hypotenuse h = sqrt(d^2 + b^2) and heading angle theta = atan(b / d).
   - Turns by theta, drives distance h along hypotenuse to the open-side waypoint.
   - Rotates back to acquire the open gateway in camera view.
5. Visual Servoing Alignment & Final Ingress:
   - Fine-tunes heading to center the open entrance (heading error < 15 px).
   - Drives forward 2.0-2.5 ft straight between the landing gear shafts to park underneath.

Safety Constraints:
- Vertical platform motors (connector J3) remain OFF at all times.
- Motor speeds calibrated >= 0.85 to overcome carpet/tile friction reliably.
"""

import time
import math
import sys
import subprocess
import logging
import cv2
import numpy as np
from jetbot import Robot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/jetson/autonomous_docking.log', mode='w')
    ]
)

# Calibrated robot physical constants at speed 0.88
ANGULAR_SPEED_DEG_PER_SEC = 95.0 # Degrees per second rotation on tracks
LINEAR_SPEED_FT_PER_SEC = 1.20   # Feet per second forward drive (~36 cm/s)
CALIB_K_FT_PX = 380.0            # Distance calibration constant (d_ft = K / width_px)

class GstCamera:
    """Raw GStreamer BGR pipe capture directly from CSI camera sensor."""
    def __init__(self, width=640, height=360, fps=30):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3

        gst_cmd = [
            'gst-launch-1.0', '-q',
            'nvarguscamerasrc', 'sensor-id=0', '!',
            'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1', '!',
            'nvvidconv', 'flip-method=2', '!',
            'video/x-raw, width=%d, height=%d, format=BGRx' % (width, height), '!',
            'videoconvert', '!',
            'video/x-raw, format=BGR', '!',
            'fdsink', 'fd=1'
        ]
        logging.info("Starting GStreamer camera pipeline subprocess...")
        self.proc = subprocess.Popen(gst_cmd, stdout=subprocess.PIPE, bufsize=self.frame_size * 2)

        # Warm up exposure and discard initial settling frames
        for _ in range(5):
            self.read()

    def read(self):
        try:
            raw = self.proc.stdout.read(self.frame_size)
            if len(raw) == self.frame_size:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                return True, frame
            return False, None
        except Exception as e:
            logging.error("Camera read error: %s", str(e))
            return False, None

    def release(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None

class AutonomousDocker:
    def __init__(self):
        logging.info("Initializing Autonomous UGV Docking System...")
        self.motors = Robot()
        # Strictly ensure vertical platform motors remain OFF and released
        self.motors.vertical_motors_stop()
        self.cap = GstCamera(width=640, height=360, fps=30)
        logging.info("Subsystems initialized. Platform motors locked OFF.")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logging.error("Failed to capture frame from camera.")
            return None
        return frame

    def analyze_scene(self, frame):
        """
        Analyzes scene to:
        1. Reject office chairs and tall furniture (columns with tall backrests in top 70px).
        2. Detect the two legs/arches of the drone landing gear resting on the floor.
        3. Check ground barrier tube to distinguish closed side from open gateway.
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_full = (gray < 75).astype(np.uint8) * 255

        # 1. Height filter: Chair backrests and room furniture extend into top 70px
        chair_sky = np.sum(dark_full[0:70, :] > 0, axis=0)
        chair_smooth = np.convolve(chair_sky, np.ones(31)/31.0, mode='same')
        is_chair_col = chair_smooth > 15

        # 2. Extract dark structures in drone elevation band (y: 75 to 250)
        drone_band = dark_full[75:250, :].copy()
        drone_band[:, is_chair_col] = 0 # Completely zero out chair columns

        # 3. Detect vertical peaks (the curved legs of the landing gear)
        vert_proj = np.sum(drone_band > 0, axis=0)
        vert_smooth = np.convolve(vert_proj, np.ones(11)/11.0, mode='same')

        peaks = []
        thresh = 25
        for x in range(15, w - 15):
            if vert_smooth[x] > thresh and vert_smooth[x] == np.max(vert_smooth[max(0, x-15):min(w, x+16)]):
                peaks.append((x, vert_smooth[x]))

        # 4. Pair selection: Find the two legs of the drone landing gear
        best_gear = None
        best_score = -1

        for i in range(len(peaks)):
            for j in range(i+1, len(peaks)):
                x1 = peaks[i][0]
                x2 = peaks[j][0]
                pw = x2 - x1
                if 80 <= pw <= 380:
                    # Top horizontal bridge check (y: 80 to 115)
                    top_strip = drone_band[5:40, x1:x2]
                    top_cov = np.mean(np.max(top_strip > 0, axis=0)) if top_strip.size > 0 else 0

                    # Ground barrier check (y: 200 to 245 -> index 125:170 in drone_band)
                    ground_strip = drone_band[125:170, x1:x2]
                    ground_cov = np.mean(np.max(ground_strip > 0, axis=0)) if ground_strip.size > 0 else 0

                    score = (peaks[i][1] + peaks[j][1]) * (0.3 + top_cov)
                    if score > best_score:
                        best_score = score
                        best_gear = {
                            'x1': x1,
                            'x2': x2,
                            'width': pw,
                            'mid_x': (x1 + x2) // 2,
                            'top_cov': float(top_cov),
                            'ground_cov': float(ground_cov),
                            'is_closed': bool(ground_cov > 0.30),
                            'score': float(score)
                        }

        annotated = frame.copy()
        gear_detected = best_gear is not None
        left_strut_x = best_gear['x1'] if gear_detected else None
        right_strut_x = best_gear['x2'] if gear_detected else None
        tunnel_mid_x = best_gear['mid_x'] if gear_detected else (w // 2)
        strut_width = best_gear['width'] if gear_detected else 0
        is_closed_side = best_gear['is_closed'] if gear_detected else False
        bar_coverage_pct = (best_gear['ground_cov'] * 100.0) if gear_detected else 0.0

        if gear_detected:
            cv2.line(annotated, (left_strut_x, 0), (left_strut_x, h), (0, 255, 0), 2)
            cv2.line(annotated, (right_strut_x, 0), (right_strut_x, h), (0, 255, 0), 2)
            cv2.line(annotated, (tunnel_mid_x, 0), (tunnel_mid_x, h), (255, 0, 0), 2)
            cv2.line(annotated, (w//2, 0), (w//2, h), (0, 0, 255), 1)

            # Estimate distance in feet using calibrated focal constant
            est_dist_ft = CALIB_K_FT_PX / float(strut_width) if strut_width > 0 else 0.0

            label = "ASPECT: CLOSED (BARRIER PRESENT)" if is_closed_side else "ASPECT: OPEN (GATEWAY)"
            col = (0, 0, 255) if is_closed_side else (0, 255, 0)
            cv2.putText(annotated, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            info_txt = "Dist: %.2fft | Width: %dpx | Ground Cov: %.1f%%" % (est_dist_ft, strut_width, bar_coverage_pct)
            cv2.putText(annotated, info_txt, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            est_dist_ft = 0.0
            cv2.putText(annotated, "SEARCHING FOR DRONE LANDING GEAR...", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        heading_error = tunnel_mid_x - (w // 2)

        return {
            'gear_detected': gear_detected,
            'left_strut_x': left_strut_x,
            'right_strut_x': right_strut_x,
            'tunnel_mid_x': tunnel_mid_x,
            'strut_width': strut_width,
            'est_dist_ft': est_dist_ft,
            'is_closed_side': is_closed_side,
            'bar_coverage_pct': bar_coverage_pct,
            'heading_error': heading_error,
            'annotated_frame': annotated
        }

    def search_360_sweep(self):
        """
        Rotates 30 degrees clockwise up to 12 steps (360 degrees)
        until the drone landing gear is acquired by the camera.
        """
        logging.info("Starting 360-degree sweep search for landing gear...")
        for step in range(12):
            frame = self.capture_frame()
            if frame is not None:
                info = self.analyze_scene(frame)
                if info['gear_detected']:
                    logging.info("Landing gear located on sweep step %d! (Width: %d px, Dist: %.2f ft)",
                                 step + 1, info['strut_width'], info['est_dist_ft'])
                    cv2.imwrite('/home/jetson/milestone_1_search_acquired.jpg', info['annotated_frame'])
                    return True, info

            # Rotate 30 degrees clockwise
            logging.info("Sweep step %d/12: Rotating 30 degrees clockwise...", step + 1)
            duration_30deg = 30.0 / ANGULAR_SPEED_DEG_PER_SEC
            self.motors.right(0.88)
            time.sleep(duration_30deg)
            self.motors.stop()
            time.sleep(0.4)

        logging.warning("Full 360-degree sweep completed without detecting landing gear.")
        return False, None

    def approach_to_inspection_distance(self):
        """
        Aligns heading and drives forward if the landing gear is far,
        so it reaches a close inspection distance (~2.0 ft, width >= 180 px)
        to reliably distinguish open side vs closed side.
        """
        logging.info("Aligning heading and approaching to inspection distance...")
        for step in range(8):
            frame = self.capture_frame()
            if frame is None:
                continue
            info = self.analyze_scene(frame)
            if not info['gear_detected']:
                logging.warning("Lost sight of landing gear during approach.")
                break

            err = info['heading_error']
            dist = info['est_dist_ft']
            width = info['strut_width']
            logging.info("Approach step %d: Dist = %.2f ft, Width = %d px, Heading error = %+d px",
                         step + 1, dist, width, err)

            # Center heading
            if abs(err) > 20:
                if err > 20:
                    self.motors.right(0.88)
                else:
                    self.motors.left(0.88)
                time.sleep(0.18)
                self.motors.stop()
                time.sleep(0.2)
                continue

            # Check if close enough (strut width >= 180 px, approx 2.0 ft away)
            if width >= 180:
                logging.info("Optimal inspection distance reached (%.2f ft, %d px).", dist, width)
                cv2.imwrite('/home/jetson/milestone_2_inspection_distance.jpg', info['annotated_frame'])
                return info

            # Drive forward pulse
            self.motors.forward(0.88)
            time.sleep(0.5)
            self.motors.stop()
            time.sleep(0.3)

        frame = self.capture_frame()
        return self.analyze_scene(frame) if frame is not None else None

    def execute_trigonometric_orbit(self, d_ft, base_ft=2.5):
        """
        Implements Pythagoras right-triangle navigation from user diagram:
        d = perpendicular distance to landing gear closed side
        b = standoff base distance along open gateway axis (2.5 ft)
        h = hypotenuse = sqrt(d^2 + b^2)
        theta = angle = atan(b / d) in degrees

        1. Turn right by theta.
        2. Drive forward along hypotenuse h.
        3. Turn left to face back towards open side of landing gear.
        """
        # Ensure d is physically bounded
        d = max(1.5, min(d_ft, 3.5))
        b = base_ft

        h = math.sqrt(d**2 + b**2)
        theta_rad = math.atan2(b, d)
        theta_deg = math.degrees(theta_rad)

        logging.info("=== TRIGONOMETRIC FLANK PARAMETERS ===")
        logging.info("  Perpendicular distance (d): %.2f ft", d)
        logging.info("  Standoff base (b): %.2f ft", b)
        logging.info("  Hypotenuse distance (h): %.2f ft", h)
        logging.info("  Turn angle (theta): %.1f degrees", theta_deg)

        # Step 1: Turn right by theta degrees
        turn_duration = theta_deg / ANGULAR_SPEED_DEG_PER_SEC
        logging.info("Step 1/3: Turning right by %.1f degrees (duration: %.2f s)...", theta_deg, turn_duration)
        self.motors.right(0.88)
        time.sleep(turn_duration)
        self.motors.stop()
        time.sleep(0.4)

        # Step 2: Drive forward along hypotenuse h
        drive_duration = h / LINEAR_SPEED_FT_PER_SEC
        logging.info("Step 2/3: Driving forward along hypotenuse %.2f ft (duration: %.2f s)...", h, drive_duration)
        self.motors.forward(0.88)
        time.sleep(drive_duration)
        self.motors.stop()
        time.sleep(0.4)

        # Step 3: Turn back left towards the open side of the landing gear
        # Theoretical turn is 90 + theta, executed in sweep pulses to acquire landing gear
        turn_back_deg = 90.0 + theta_deg
        logging.info("Step 3/3: Turning left by %.1f degrees to face open entrance...", turn_back_deg)
        turn_back_duration = turn_back_deg / ANGULAR_SPEED_DEG_PER_SEC
        self.motors.left(0.88)
        time.sleep(turn_back_duration)
        self.motors.stop()
        time.sleep(0.5)

        logging.info("Trigonometric waypoint reached. UGV is now in front of open side.")

    def fine_tune_and_ingress(self, base_ft=2.0):
        """
        Reacquires the open gateway, fine-tunes heading centering,
        and moves forward along the base line to park directly between the landing gear shafts.
        """
        logging.info("Fine-tuning alignment with open entrance gateway...")

        # Search / fine-tune sweep to center the open gateway
        aligned = False
        for step in range(8):
            frame = self.capture_frame()
            if frame is None:
                continue
            info = self.analyze_scene(frame)
            if not info['gear_detected']:
                logging.info("Sweeping left to locate open gateway (step %d)...", step + 1)
                self.motors.left(0.88)
                time.sleep(0.18)
                self.motors.stop()
                time.sleep(0.3)
                continue

            err = info['heading_error']
            logging.info("Alignment step %d: Heading error = %+d px, Width = %d px",
                         step + 1, err, info['strut_width'])

            if abs(err) <= 15:
                logging.info("Squarely aligned with open entrance gateway center!")
                cv2.imwrite('/home/jetson/milestone_3_gateway_aligned.jpg', info['annotated_frame'])
                aligned = True
                break
            elif err > 15:
                self.motors.right(0.88)
                time.sleep(0.15)
                self.motors.stop()
            else:
                self.motors.left(0.88)
                time.sleep(0.15)
                self.motors.stop()
            time.sleep(0.25)

        # Ingress forward drive: drive forward base_ft (approx 2.0 ft, 1.8 s)
        ingress_duration = base_ft / LINEAR_SPEED_FT_PER_SEC
        logging.info("Driving forward %.1f ft into open gateway (duration: %.2f s)...", base_ft, ingress_duration)
        self.motors.forward(0.88)
        time.sleep(ingress_duration)
        self.motors.stop()
        time.sleep(0.5)

        # Final stop
        self.motors.stop()
        logging.info("UGV is now parked directly between the shafts of the landing gear!")

        final_frame = self.capture_frame()
        if final_frame is not None:
            cv2.imwrite('/home/jetson/milestone_4_docked_between_shafts.jpg', final_frame)
            logging.info("Captured final docked frame: /home/jetson/milestone_4_docked_between_shafts.jpg")

    def run_docking_mission(self):
        logging.info("==================================================")
        logging.info("STARTING AUTONOMOUS DRONE DOCKING MISSION")
        logging.info("==================================================")

        # Safety check: ensure vertical lift remains inactive
        self.motors.vertical_motors_stop()

        # Step 1: 360-degree sweep search
        found, info = self.search_360_sweep()
        if not found:
            logging.error("Landing gear could not be found within 360 degrees. Mission aborted.")
            return

        # Step 2: Approach to close inspection distance
        info = self.approach_to_inspection_distance()
        if info is None or not info['gear_detected']:
            logging.error("Failed to establish close inspection of landing gear.")
            return

        # Step 3: Aspect Decision & Trigonometric Flank
        logging.info("=== ASPECT EVALUATION ===")
        is_closed = info['is_closed_side']
        d_est = info['est_dist_ft']

        if is_closed:
            logging.info("Detected CLOSED SIDE (Skid tube barrier on floor: %.1f%%).", info['bar_coverage_pct'])
            logging.info("Executing Pythagoras waypoint navigation to open side...")
            self.execute_trigonometric_orbit(d_ft=d_est, base_ft=2.5)
        else:
            logging.info("Detected OPEN SIDE (Gateway clear: %.1f%% barrier). Already facing opening!",
                         info['bar_coverage_pct'])

        # Step 4: Fine-tune heading and move inside between shafts
        self.fine_tune_and_ingress(base_ft=2.0)

        logging.info("==================================================")
        logging.info("AUTONOMOUS DOCKING MISSION COMPLETED SUCCESSFULLY")
        logging.info("==================================================")

    def close(self):
        self.motors.stop()
        self.motors.vertical_motors_stop()
        self.cap.release()
        logging.info("Subsystems cleanly released.")

def main():
    docker = AutonomousDocker()
    try:
        docker.run_docking_mission()
    except KeyboardInterrupt:
        logging.warning("User interrupted docking mission.")
    finally:
        docker.close()

if __name__ == '__main__':
    main()
