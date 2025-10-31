# Posture Monitor Application (app.py)
# Toàn bộ mã nguồn hoàn chỉnh của chương trình giám sát tư thế bằng MediaPipe và OpenCV:
# Ref: https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/

import cv2
import math as m
import mediapipe as mp
import argparse
import sys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def findDistance(x1, y1, x2, y2):
    """Tính khoảng cách Euclid giữa hai điểm."""
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    """Tính góc giữa hai điểm với trục y."""
    y1s = y1 if y1 != 0 else 1e-6
    denom = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1s
    denom = denom if denom != 0 else 1e-6
    theta = m.acos((y2 - y1) * (-y1s) / denom)
    return int(180 / m.pi * theta)

def sendWarning():
    """Hàm cảnh báo tư thế xấu quá lâu."""
    print("⚠️  Cảnh báo: Tư thế xấu quá lâu!")

def parse_arguments():
    """Parse các tham số khi chạy chương trình."""
    p = argparse.ArgumentParser(description="Posture Monitor with MediaPipe")
    p.add_argument("--video", type=str, default="0", help="Đường dẫn video, hoặc số index webcam (vd: 0).")
    p.add_argument("--offset-threshold", type=int, default=100, help="Ngưỡng cân vai (px).")
    p.add_argument("--neck-angle-threshold", type=int, default=25, help="Ngưỡng góc cổ (độ).")
    p.add_argument("--torso-angle-threshold", type=int, default=10, help="Ngưỡng góc thân (độ).")
    p.add_argument("--time-threshold", type=int, default=180, help="Ngưỡng thời gian tư thế xấu để cảnh báo (s).")
    p.add_argument("--show", action="store_true", help="Hiển thị cửa sổ (dùng khi chạy local).")
    p.add_argument("--output", type=str, default="output.mp4", help="Tên file video kết quả.")
    return p.parse_args()

def main(video_path, offset_threshold=100, neck_angle_threshold=25, torso_angle_threshold=10, time_threshold=180, show=False, output_path="output.mp4"):
    good_frames, bad_frames = 0, 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
    white = (255, 255, 255)

    # Mở video hoặc webcam
    if str(video_path).isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không mở được video/camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 1 else 30.0, (width, height))

    pose = mp_pose.Pose()

    while True:
        ok, image = cap.read()
        if not ok:
            print("🔹 Không còn khung hình (hết video hoặc lỗi camera).")
            break

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]

        if not result.pose_landmarks:
            out.write(image)
            if show:
                cv2.imshow("Posture", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        lm = result.pose_landmarks.landmark
        lmPose = mp_pose.PoseLandmark

        # Lấy các điểm quan trọng
        l_shldr_x = int(lm[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm[lmPose.LEFT_HIP].y * h)

        # Tính offset vai
        offset = int(findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y))
        right_text = f"{offset} Shoulders aligned" if offset < offset_threshold else f"{offset} Shoulders not aligned"
        right_color = green if offset < offset_threshold else red
        cv2.putText(image, right_text, (w - 280, 30), font, 0.6, right_color, 2)

        # Tính góc cổ và thân
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Vẽ các landmark
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, white, 2)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, white, 2)
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, white, 2)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Kiểm tra tư thế tốt hay xấu
        good_posture = (neck_inclination < neck_angle_threshold and torso_inclination < torso_angle_threshold)
        if good_posture:
            bad_frames = 0
            good_frames += 1
            line_color = green
            text_color = light_green
        else:
            good_frames = 0
            bad_frames += 1
            line_color = red
            text_color = red

        # Vẽ các đường nối
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), line_color, 2)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), line_color, 2)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), line_color, 2)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), line_color, 2)

        # Hiển thị góc và thời gian tư thế
        cv2.putText(image, f"Neck inclination: {int(neck_inclination)}", (10, 20), font, 0.6, text_color, 2)
        cv2.putText(image, f"Torso inclination: {int(torso_inclination)}", (10, 45), font, 0.6, text_color, 2)
        _fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
        good_time = (1.0 / _fps) * good_frames
        bad_time = (1.0 / _fps) * bad_frames

        if good_time > 0:
            cv2.putText(image, f"Good Posture Time : {round(good_time,1)}s", (10, h - 15), font, 0.7, green, 2)
        else:
            cv2.putText(image, f"Bad Posture Time : {round(bad_time,1)}s", (10, h - 15), font, 0.7, red, 2)

        if bad_time > time_threshold:
            sendWarning()
            cv2.putText(image, "BAD POSTURE TOO LONG!", (int(w*0.25), int(h*0.1)), font, 1.0, red, 3)

        # Ghi ra file
        out.write(image)

        # Hiển thị màn hình (nếu bật --show)
        if show:
            cv2.imshow("Posture", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    print(f"✅ Xong! Video kết quả: {output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    print("Arguments:")
    print(f"Video: {args.video}")
    print(f"Offset Threshold: {args.offset_threshold}")
    print(f"Neck Angle Threshold: {args.neck_angle_threshold}")
    print(f"Torso Angle Threshold: {args.torso_angle_threshold}")
    print(f"Time Threshold: {args.time_threshold}")
    main(
        video_path=args.video,
        offset_threshold=args.offset_threshold,
        neck_angle_threshold=args.neck_angle_threshold,
        torso_angle_threshold=args.torso_angle_threshold,
        time_threshold=args.time_threshold,
        show=args.show,
        output_path=args.output
    )
