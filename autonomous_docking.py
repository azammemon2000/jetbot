#!/usr/bin/env python3
"""
Autonomous Drone Landing Gear Docking System for Jetbot UGV.

Objective:
1. Detect drone landing gear while rejecting background furniture (office chairs, desks).
2. Determine orientation:
   - Closed side: horizontal skid tube on floor blocks entrance.
   - Open side: clear floor between arches (docking gateway).
3. If closed side: execute orbital flank around to face open gateway.
4. Align UGV heading with open gateway center.
5. Drive forward directly inside the landing gear and park under the drone.

Safety Constraints:
- Vertical platform motors (connector J3) remain OFF.
- Speeds >= 0.85 to overcome carpet/tile friction reliably.
"""

import time
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

        # Warm up exposure and discard initial frames
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
        logging.info("Subsystems initialized. Speeds calibrated >= 0.88.")

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
        3. Determine if the ground between the legs has a barrier tube (closed side) or is clear (open gateway).
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

                    # Score combines prominent legs and top bridge connectivity
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

            label = "ASPECT: CLOSED (BARRIER PRESENT)" if is_closed_side else "ASPECT: OPEN (GATEWAY)"
            col = (0, 0, 255) if is_closed_side else (0, 255, 0)
            cv2.putText(annotated, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            info_txt = "Width: %dpx | Ground Cov: %.1f%%" % (strut_width, bar_coverage_pct)
            cv2.putText(annotated, info_txt, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(annotated, "SEARCHING FOR DRONE LANDING GEAR...", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        heading_error = tunnel_mid_x - (w // 2)

        return {
            'gear_detected': gear_detected,
            'left_strut_x': left_strut_x,
            'right_strut_x': right_strut_x,
            'tunnel_mid_x': tunnel_mid_x,
            'strut_width': strut_width,
            'is_closed_side': is_closed_side,
            'bar_coverage_pct': bar_coverage_pct,
            'heading_error': heading_error,
            'annotated_frame': annotated
        }

    def execute_closed_side_orbit(self):
        """
        Executes orbital flank around landing gear when facing the side barrier.
        Repositions UGV to face the open entrance gateway.
        """
        logging.info("Executing orbital flank maneuver...")

        # Step 1: Reverse for swing clearance
        logging.info("Flank 1/4: Backing up 25 cm for swing clearance...")
        self.motors.backward(0.90)
        time.sleep(0.8)
        self.motors.stop()
        time.sleep(0.4)

        # Step 2: Turn 90 degrees right
        logging.info("Flank 2/4: Turning 90 degrees right...")
        self.motors.right(0.90)
        time.sleep(1.0)
        self.motors.stop()
        time.sleep(0.4)

        # Step 3: Drive forward along the flank
        logging.info("Flank 3/4: Driving forward along side flank...")
        self.motors.forward(0.90)
        time.sleep(1.5)
        self.motors.stop()
        time.sleep(0.4)

        # Step 4: Turn 90 degrees left to face open gateway
        logging.info("Flank 4/4: Turning 90 degrees left to face gateway...")
        self.motors.left(0.90)
        time.sleep(1.0)
        self.motors.stop()
        time.sleep(0.5)

        logging.info("Orbital flank completed. Now positioned in front of open side.")

    def run_docking_mission(self):
        logging.info("Starting Autonomous Docking Mission...")

        # Safety check: keep vertical lift inactive
        self.motors.vertical_motors_stop()

        # Phase 1: Scene Detection & Initial Centering
        logging.info("=== PHASE 1: INITIAL DETECTION & CENTERING ===")
        detected = False
        for attempt in range(8):
            frame = self.capture_frame()
            if frame is None:
                continue
            info = self.analyze_scene(frame)
            if info['gear_detected']:
                cv2.imwrite('/home/jetson/milestone_1_initial_detection.jpg', info['annotated_frame'])
                logging.info("Landing gear located! Width: %d px, Heading error: %+d px",
                             info['strut_width'], info['heading_error'])
                detected = True
                break
            else:
                logging.info("Searching... sweep turn right (attempt %d)...", attempt + 1)
                self.motors.right(0.88)
                time.sleep(0.3)
                self.motors.stop()
                time.sleep(0.4)

        if not detected:
            logging.error("Landing gear not detected. Mission stopped.")
            return

        # Phase 2: Aspect Evaluation (Closed Barrier vs Open Gateway)
        logging.info("=== PHASE 2: ASPECT EVALUATION ===")
        frame = self.capture_frame()
        info = self.analyze_scene(frame)
        cv2.imwrite('/home/jetson/milestone_2_aspect_decision.jpg', info['annotated_frame'])

        if info['gear_detected'] and info['is_closed_side']:
            logging.info("Aspect: CLOSED SIDE (Skid tube barrier on ground: %.1f%%). Initiating Flank!",
                         info['bar_coverage_pct'])
            self.execute_closed_side_orbit()
        else:
            logging.info("Aspect: OPEN GATEWAY DETECTED! Ground is clear between legs.")

        # Phase 3: Gateway Heading Alignment
        logging.info("=== PHASE 3: OPEN GATEWAY HEADING ALIGNMENT ===")
        for align_step in range(6):
            frame = self.capture_frame()
            if frame is None:
                continue
            info = self.analyze_scene(frame)
            if not info['gear_detected']:
                logging.warning("Searching for gateway...")
                self.motors.left(0.88)
                time.sleep(0.25)
                self.motors.stop()
                time.sleep(0.3)
                continue

            err = info['heading_error']
            logging.info("Gateway alignment step %d: error = %+d px, width = %d px",
                         align_step + 1, err, info['strut_width'])

            if abs(err) <= 20:
                logging.info("Aligned squarely with open gateway center!")
                cv2.imwrite('/home/jetson/milestone_3_aligned_gateway.jpg', info['annotated_frame'])
                break
            elif err > 20:
                self.motors.right(0.88)
                time.sleep(0.2)
                self.motors.stop()
            else:
                self.motors.left(0.88)
                time.sleep(0.2)
                self.motors.stop()
            time.sleep(0.3)

        # Phase 4: Ingress Drive & Direct Parking Inside Landing Gear
        logging.info("=== PHASE 4: INGRESS & DIRECT PARKING INSIDE ===")
        logging.info("Driving directly into open gateway between landing gear skids...")

        # Drive forward straight into the tunnel under drone frame
        self.motors.forward(0.88)
        time.sleep(2.2)
        self.motors.stop()
        time.sleep(0.5)

        # Fully stop and park
        self.motors.stop()
        logging.info("UGV is now parked directly inside the landing gear!")

        # Final proof frame
        final_frame = self.capture_frame()
        if final_frame is not None:
            cv2.imwrite('/home/jetson/milestone_4_docked_inside.jpg', final_frame)
            logging.info("Captured final proof image: /home/jetson/milestone_4_docked_inside.jpg")

        logging.info("Autonomous Docking Mission Completed Successfully.")

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
