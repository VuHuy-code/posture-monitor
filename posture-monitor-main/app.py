# ============================================================
# Posture Monitor v3.6 (with Pygame Sound + Timed Audio in Output)
# Author: GPT-5 for Huy
# Features:
# - Detects 13 keypoints (ears, shoulders, elbows, hips, wrists, nose)
# - Identifies posture issues: forward head, slouching, uneven shoulders, arm support
# - Gives English feedback and correction tips
# - Works on Windows / macOS / Linux (even in headless mode)
# - Plays live beep when bad posture persists
# - Automatically inserts beep sounds into the exported video at exact bad posture times
# ============================================================

import cv2
import math
import mediapipe as mp
from collections import deque
import argparse
import os
import numpy as np
import wave
import struct
import pygame
import subprocess

# ===== SOUND HANDLING =====
def ensure_alert_sound():
    """Create alert.wav if it doesn't exist"""
    if not os.path.exists("alert.wav"):
        framerate = 44100
        duration = 0.4
        frequency = 800
        n_samples = int(framerate * duration)
        t = np.linspace(0, duration, n_samples)
        data = np.sin(2 * np.pi * frequency * t)
        data = (data * 32767).astype(np.int16)
        with wave.open("alert.wav", "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(framerate)
            f.writeframes(struct.pack('<' + ('h' * len(data)), *data))

def beep():
    """Play alert sound using pygame"""
    ensure_alert_sound()
    try:
        pygame.mixer.init()
        sound = pygame.mixer.Sound("alert.wav")
        sound.play()
        print("🔊 Beep sound played.")
    except Exception as e:
        print("⚠️ Cannot play sound:", e)

# ===== POSE DETECTION =====
mp_pose = mp.solutions.pose
SMOOTH_WINDOW = 5

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default="test.mp4", help="Path to video file or webcam index (0).")
    p.add_argument("--time-threshold", type=float, default=10, help="Seconds before alarm triggers.")
    p.add_argument("--show", action="store_true", help="Show live preview (needs GUI).")
    p.add_argument("--output", type=str, default="output.mp4", help="Output video file.")
    return p.parse_args()

def mid(a, b): return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
def vec(a, b): return (b[0] - a[0], b[1] - a[1])
def dist(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
    if mag == 0: return 0
    return math.degrees(math.acos(max(min(dot/mag, 1), -1)))

def angle_with_vertical(v):
    dx, dy = v
    ang = abs(math.degrees(math.atan2(dx, -dy)))
    return min(ang, 180)

def get_point(lm, idx, w, h):
    p = lm[idx]
    return (p.x * w, p.y * h), p.visibility

# ===== MAIN FUNCTION =====
def main():
    args = parse_args()
    cap = cv2.VideoCapture(int(args.video) if str(args.video).isdigit() else args.video)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    neck_buf, torso_buf = deque(maxlen=SMOOTH_WINDOW), deque(maxlen=SMOOTH_WINDOW)
    bad_frames, good_frames = 0, 0
    alarm_triggered = False
    bad_timestamps = []  # Save times when bad posture detected

    font = cv2.FONT_HERSHEY_SIMPLEX
    C = {"g": (0,255,0), "r": (0,0,255), "y": (0,255,255), "w": (255,255,255), "b": (255,0,0)}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if not res.pose_landmarks:
            out.write(frame)
            continue

        lm = res.pose_landmarks.landmark
        L = mp_pose.PoseLandmark
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

        mid_sh = mid(pts["LShoulder"], pts["RShoulder"])
        mid_ear = mid(pts["LEar"], pts["REar"])
        mid_hip = mid(pts["LHip"], pts["RHip"])
        neck = mid(mid_sh, mid_ear)

        neck_vec = vec(mid_sh, mid_ear)
        torso_vec = vec(mid_hip, mid_sh)
        spine_vec = vec(mid_hip, mid_ear)

        neck_angle = angle_with_vertical(neck_vec)
        torso_angle = angle_with_vertical(torso_vec)

        neck_buf.append(neck_angle)
        torso_buf.append(torso_angle)
        s_neck = sum(neck_buf)/len(neck_buf)
        s_torso = sum(torso_buf)/len(torso_buf)

        shoulder_diff = abs(pts["LShoulder"][1] - pts["RShoulder"][1])
        shoulder_imbalance = shoulder_diff > 40
        ear_shoulder_dist = dist(mid_ear, mid_sh)
        fwd_head = ear_shoulder_dist > (0.22 * w)
        spine_angle = angle_with_vertical(spine_vec)
        slouch = spine_angle > 20

        l_elbow_angle = angle(vec(pts["LShoulder"], pts["LElbow"]), vec(pts["LElbow"], pts["LWrist"]))
        r_elbow_angle = angle(vec(pts["RShoulder"], pts["RElbow"]), vec(pts["RElbow"], pts["RWrist"]))
        arm_support = l_elbow_angle < 50 or r_elbow_angle < 50

        issues, tips = [], []
        if fwd_head:
            issues.append("Forward Head")
            tips.append("Keep your head aligned with shoulders.")
        if slouch:
            issues.append("Slouching")
            tips.append("Straighten your back and open your chest.")
        if shoulder_imbalance:
            issues.append("Uneven Shoulders")
            tips.append("Keep both shoulders relaxed and even.")
        if arm_support:
            issues.append("Leaning on Arm")
            tips.append("Balance your arms equally on the desk.")

        bad_posture = len(issues) > 0
        fps_ = fps if fps > 1 else 30
        if bad_posture:
            bad_frames += 1
            good_frames = 0
            color = C["r"]
        else:
            good_frames += 1
            bad_frames = 0
            color = C["g"]

        bad_time = bad_frames / fps_
        if bad_time > args.time_threshold and not alarm_triggered:
            print("🔊 ALERT: Bad posture detected for too long!")
            beep()
            alarm_triggered = True
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            bad_timestamps.append(round(current_time, 2))
        if good_frames > 10:
            alarm_triggered = False

        # Draw keypoints and guide lines
        for name in pts:
            cv2.circle(frame, tuple(map(int, pts[name])), 4, C["w"], -1)
        for a, b in [
            ("LShoulder", "RShoulder"), ("LEar", "REar"),
            ("LHip", "RHip"), ("LShoulder", "LElbow"),
            ("RShoulder", "RElbow"), ("LElbow", "LWrist"),
            ("RElbow", "RWrist"), ("LShoulder", "LHip"),
            ("RShoulder", "RHip")
        ]:
            cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), C["y"], 2)

        ideal_sh_y = int((pts["LShoulder"][1] + pts["RShoulder"][1]) / 2)
        ideal_head_x = int((pts["LShoulder"][0] + pts["RShoulder"][0]) / 2)
        cv2.line(frame, (ideal_head_x, ideal_sh_y - 150), (ideal_head_x, ideal_sh_y + 200), C["g"], 2)
        cv2.line(frame, (int(w*0.3), ideal_sh_y), (int(w*0.7), ideal_sh_y), C["g"], 2)

        cv2.putText(frame, f"Neck: {s_neck:.1f}°  Torso: {s_torso:.1f}°", (10, 25), font, 0.6, color, 2)
        if bad_posture:
            y0 = 60
            for i, txt in enumerate(issues):
                cv2.putText(frame, f"- {txt}", (10, y0 + 25*i), font, 0.7, C["r"], 2)
            y1 = y0 + 25 * len(issues) + 10
            for i, tip in enumerate(tips):
                cv2.putText(frame, f"Tip: {tip}", (10, y1 + 25*i), font, 0.55, C["y"], 2)
        else:
            cv2.putText(frame, "✅ Good posture. Keep it up!", (10, 55), font, 0.7, C["g"], 2)

        out.write(frame)
        if args.show:
            try:
                cv2.imshow("Posture Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                pass

    cap.release()
    out.release()

    # ===== INSERT BEEP SOUND EXACTLY AT BAD POSTURE TIMES =====
    try:
        merged_output = "output_with_beeps.mp4"
        if bad_timestamps:
            print("🎵 Inserting beep sounds at these times:", bad_timestamps)
            cmd_list = ["ffmpeg", "-y", "-i", args.output]
            filter_complex = ""
            for i, t in enumerate(bad_timestamps):
                cmd_list += ["-i", "alert.wav"]
                filter_complex += f"[{i+1}:a]adelay={int(t*1000)}|{int(t*1000)}[a{i}];"
            filter_complex += "".join([f"[a{i}]" for i in range(len(bad_timestamps))])
            filter_complex += f"amix=inputs={len(bad_timestamps)}[outa]"
            cmd_list += [
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[outa]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                merged_output
            ]
            subprocess.run(cmd_list, check=True)
            print(f"🎬 Final video with beeps saved as: {merged_output}")
        else:
            print("✅ No bad posture detected — no beep inserted.")
    except Exception as e:
        print("⚠️ Could not merge audio:", e)

    print(f"✅ Done! Output saved to {args.output}")

if __name__ == "__main__":
    main()
