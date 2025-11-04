# Posture Monitor - Full Upper Body Detection v3.1
# Author: GPT-5 for Nguyễn Đình Quân
# Features:
# - Detect 13 keypoints: ears, shoulders, elbows, hips, wrists, nose
# - Identify common posture issues: forward head, shoulder imbalance, slouching, leaning, arm support
# - Multi-angle and smoothed output
# - Visual skeleton overlay

import cv2
import math
import mediapipe as mp
from collections import deque
import argparse

mp_pose = mp.solutions.pose
SMOOTH_WINDOW = 5  # smoothing frame window

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default="0", help="Path to video file or webcam index (0).")
    p.add_argument("--time-threshold", type=float, default=30)
    p.add_argument("--show", action="store_true")
    p.add_argument("--output", type=str, default="output.mp4")
    return p.parse_args()

def mid(a, b): return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
def vec(a, b): return (b[0] - a[0], b[1] - a[1])
def dist(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
    if mag == 0: return 0
    ang = math.degrees(math.acos(max(min(dot / mag, 1), -1)))
    return ang

def angle_with_vertical(v):
    dx, dy = v
    ang = abs(math.degrees(math.atan2(dx, -dy)))
    return min(ang, 180)

def get_point(lm, idx, w, h):
    p = lm[idx]
    return (p.x * w, p.y * h), p.visibility

def main():
    args = parse_args()
    cap = cv2.VideoCapture(int(args.video) if str(args.video).isdigit() else args.video)
    if not cap.isOpened():
        print("❌ Không mở được video.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    neck_buf, torso_buf = deque(maxlen=SMOOTH_WINDOW), deque(maxlen=SMOOTH_WINDOW)
    bad_frames, good_frames = 0, 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    C = {"g": (0, 255, 0), "r": (0, 0, 255), "y": (0, 255, 255), "w": (255, 255, 255)}

    while True:
        ok, frame = cap.read()
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if not res.pose_landmarks:
            out.write(frame)
            continue

        lm = res.pose_landmarks.landmark
        L = mp_pose.PoseLandmark

        # --- Get 13 keypoints ---
        pts = {}
        for name, idx in [
            ("LEar", L.LEFT_EAR), ("REar", L.RIGHT_EAR),
            ("LShoulder", L.LEFT_SHOULDER), ("RShoulder", L.RIGHT_SHOULDER),
            ("LElbow", L.LEFT_ELBOW), ("RElbow", L.RIGHT_ELBOW),
            ("LWrist", L.LEFT_WRIST), ("RWrist", L.RIGHT_WRIST),
            ("LHip", L.LEFT_HIP), ("RHip", L.RIGHT_HIP),
            ("Nose", L.NOSE)
        ]:
            pts[name], _ = get_point(lm, idx, w, h)

        # --- Compute reference midpoints ---
        mid_sh = mid(pts["LShoulder"], pts["RShoulder"])
        mid_ear = mid(pts["LEar"], pts["REar"])
        mid_hip = mid(pts["LHip"], pts["RHip"])
        neck = mid(mid_sh, mid_ear)

        # --- Angles ---
        neck_vec = vec(mid_sh, mid_ear)
        torso_vec = vec(mid_hip, mid_sh)
        spine_vec = vec(mid_hip, mid_ear)

        neck_angle = angle_with_vertical(neck_vec)
        torso_angle = angle_with_vertical(torso_vec)

        neck_buf.append(neck_angle)
        torso_buf.append(torso_angle)
        s_neck = sum(neck_buf) / len(neck_buf)
        s_torso = sum(torso_buf) / len(torso_buf)

        # --- Posture issues detection ---
        shoulder_diff = abs(pts["LShoulder"][1] - pts["RShoulder"][1])
        shoulder_imbalance = shoulder_diff > 40

        ear_shoulder_dist = dist(mid_ear, mid_sh)
        fwd_head = ear_shoulder_dist > (0.22 * w)

        spine_angle = angle_with_vertical(spine_vec)
        slouch = spine_angle > 20

        l_elbow_angle = angle(vec(pts["LShoulder"], pts["LElbow"]), vec(pts["LElbow"], pts["LWrist"]))
        r_elbow_angle = angle(vec(pts["RShoulder"], pts["RElbow"]), vec(pts["RElbow"], pts["RWrist"]))
        arm_support = l_elbow_angle < 50 or r_elbow_angle < 50

        issues = []
        if fwd_head: issues.append("Head forward")
        if slouch: issues.append("Slouching")
        if shoulder_imbalance: issues.append("Uneven shoulders")
        if arm_support: issues.append("Arm support")
        bad_posture = len(issues) > 0

        # --- Posture logic ---
        if bad_posture:
            bad_frames += 1
            good_frames = 0
            color = C["r"]
        else:
            good_frames += 1
            bad_frames = 0
            color = C["g"]

        # --- Draw skeleton ---
        for name in pts:
            cv2.circle(frame, tuple(map(int, pts[name])), 4, C["w"], -1)

        for a, b in [
            ("LShoulder", "RShoulder"),
            ("LEar", "REar"),
            ("LHip", "RHip"),
            ("LShoulder", "LElbow"),
            ("RShoulder", "RElbow"),
            ("LElbow", "LWrist"),
            ("RElbow", "RWrist"),
            ("LShoulder", "LHip"),
            ("RShoulder", "RHip")
        ]:
            cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), C["y"], 2)

        # --- Text display ---
        cv2.putText(frame, f"Neck: {s_neck:.1f}°  Torso: {s_torso:.1f}°", (10, 25), font, 0.6, color, 2)
        fps_ = fps if fps > 1 else 30
        good_time, bad_time = good_frames / fps_, bad_frames / fps_

        if bad_time > args.time_threshold:
            cv2.putText(frame, "⚠️ BAD POSTURE TOO LONG!", (int(w * 0.25), int(h * 0.1)), font, 1.0, C["r"], 3)

        if bad_posture:
            y0 = 55
            for i, txt in enumerate(issues):
                cv2.putText(frame, f"- {txt}", (10, y0 + 25 * i), font, 0.6, C["r"], 2)
        else:
            cv2.putText(frame, "Good posture", (10, 55), font, 0.7, C["g"], 2)

        out.write(frame)
        if args.show:
            cv2.imshow("Posture Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    print(f"✅ Done! Output saved to {args.output}")

if __name__ == "__main__":
    main()
