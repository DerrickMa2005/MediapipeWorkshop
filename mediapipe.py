import argparse
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# -------- Config (tweak if needed) --------
# These are normalized thresholds and visibility filters.
MIN_VIS = 0.6            # how confident a landmark must be to use
HANDS_ABOVE_MARGIN = 0.05 # how much above nose counts as "above head"
FEET_APART_FACTOR = 1.15  # how many times hip width ankles must be apart to count "open"
EMA_ALPHA = 0.3           # smoothing for distances (0=no smoothing, 1=no smoothing too; use ~0.2-0.4)

# ---------- Helpers ----------
PL = mp.solutions.pose.PoseLandmark

def get_xy(landmarks, idx):
    lm = landmarks[idx]
    return lm.x, lm.y, lm.visibility

def good(*vis_list):
    return all(v is not None and v >= MIN_VIS for v in vis_list)

def ema(prev, new, alpha=EMA_ALPHA):
    if prev is None:
        return new
    return alpha*new + (1 - alpha)*prev

# ---------- Jumping jack logic ----------
class JJCounter:
    """
    Counts a rep when we see CLOSED -> OPEN -> CLOSED.
    CLOSED: hands not above head OR feet not apart enough.
    OPEN:   hands above head AND feet clearly apart.
    """
    def __init__(self):
        self.state = "CLOSED"
        self.was_open = False
        self.count = 0

        # smoothed measures
        self._ankle_sep_s = None
        self._hip_width_s = None

        # simple FPS
        self._t0 = time.time()
        self._frames = deque(maxlen=15)
        self.fps = 0.0

    def update_fps(self):
        t = time.time()
        self._frames.append(t)
        if len(self._frames) >= 2:
            dt = self._frames[-1] - self._frames[0]
            self.fps = (len(self._frames)-1) / dt if dt > 0 else 0.0

    def step(self, landmarks):
        """
        landmarks: normalized (x, y, visibility) from MediaPipe world.
        """
        # Required points
        nose = get_xy(landmarks, PL.NOSE)
        l_wrist = get_xy(landmarks, PL.LEFT_WRIST)
        r_wrist = get_xy(landmarks, PL.RIGHT_WRIST)
        l_elbow = get_xy(landmarks, PL.LEFT_ELBOW)
        r_elbow = get_xy(landmarks, PL.RIGHT_ELBOW)
        l_hip = get_xy(landmarks, PL.LEFT_HIP)
        r_hip = get_xy(landmarks, PL.RIGHT_HIP)
        l_ankle = get_xy(landmarks, PL.LEFT_ANKLE)
        r_ankle = get_xy(landmarks, PL.RIGHT_ANKLE)

        # Visibility checks
        if not good(nose[2], l_wrist[2], r_wrist[2], l_elbow[2], r_elbow[2], l_hip[2], r_hip[2], l_ankle[2], r_ankle[2]):
            return self.count, self.state, {"hands_above": False, "feet_apart": False}

        # Hands above head:
        # In normalized image coords, y increases downward.
        # So "above head" means y_wrist < y_nose - margin (i.e., smaller y).
        y_nose = nose[1]
        y_wrists_min = min(l_wrist[1], r_wrist[1], l_elbow[1], r_elbow[1])
        hands_above = y_wrists_min < (y_nose - HANDS_ABOVE_MARGIN)

        # Feet apart vs hip width (scale-invariant)
        hip_width = abs(l_hip[0] - r_hip[0])
        ankle_sep = abs(l_ankle[0] - r_ankle[0])

        # Smooth these (helps jitter)
        self._hip_width_s  = ema(self._hip_width_s,  hip_width)
        self._ankle_sep_s  = ema(self._ankle_sep_s,  ankle_sep)

        # Guard against division by tiny hip width
        feet_apart = False
        if self._hip_width_s and self._hip_width_s > 1e-3:
            feet_apart = (self._ankle_sep_s / self._hip_width_s) >= FEET_APART_FACTOR

        is_open = hands_above and feet_apart

        # State machine
        if self.state == "CLOSED":
            if is_open:
                self.state = "OPEN"
                self.was_open = True
        else:  # OPEN
            if not is_open:
                # completed one full open->closed cycle
                if self.was_open:
                    self.count += 1
                    self.was_open = False
                self.state = "CLOSED"

        return self.count, self.state, {"hands_above": hands_above, "feet_apart": feet_apart}

# ---------- Main video loop ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index (e.g., 0) or video file path")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--mirror", action="store_true", help="Flip frame horizontally for selfie view")
    args = parser.parse_args()

    src = 0
    if args.source != "0":
        # try int, else assume path
        try:
            src = int(args.source)
        except ValueError:
            src = args.source

    cap = cv2.VideoCapture(src)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)
    drawer = mp.solutions.drawing_utils
    style = mp.solutions.drawing_styles

    counter = JJCounter()

    window = "Jumping Jack Counter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                # Count logic
                count, state, flags = counter.step(res.pose_landmarks.landmark)

                # Draw skeleton
                drawer.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=style.get_default_pose_landmarks_style())

                # Overlay info
                h, w = frame.shape[:2]
                pad = 16
                cv2.rectangle(frame, (pad-6, pad-6), (300, pad+82), (0,0,0), -1)
                cv2.putText(frame, f"JJs: {count}", (pad, pad+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"State: {state}", (pad, pad+45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Hands:{'UP ' if flags['hands_above'] else 'down'}  Feet:{'APART' if flags['feet_apart'] else 'together'}",
                            (pad, pad+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,255), 1, cv2.LINE_AA)

            # FPS
            counter.update_fps()
            cv2.putText(frame, f"{counter.fps:4.1f} FPS", (frame.shape[1]-140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()
