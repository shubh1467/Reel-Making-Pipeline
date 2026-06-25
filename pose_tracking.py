import cv2
import csv
import numpy as np
import os

from mediapipe.python.solutions import pose as mp_pose


pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# ==================================
# SPEED CONTROLS
# ==================================

INPUT_DUPLICATE_FACTOR = 1    # Original Speed = 1, 2 = 0.5x speed, 3 = 0.33x speed, etc.
OUTPUT_SPEED_FACTOR = 0.50    # 0.50 = output video plays at half speed

# ==================================
# EXTRA: Save raw input slowed to 0.5x
# ==================================

SAVE_RAW_SLOWED_VIDEO = True   # Set to True to also save the original video at 0.5x speed
RAW_SLOWED_OUTPUT = "raw_slowed_0.5x_mediapipe.mp4"

# ==================================
# EMA parameters
# ==================================

ALPHA = 0.20
smooth_cx = None

# ==================================
# Mapping MediaPipe landmarks to COCO keypoints
# ==================================


MP_TO_COCO = {
    0: 0,      # nose
    11: 5,     # left_shoulder
    12: 6,     # right_shoulder
    13: 7,     # left_elbow
    14: 8,     # right_elbow
    15: 9,     # left_wrist
    16: 10,    # right_wrist
    23: 11,    # left_hip
    24: 12,    # right_hip
    25: 13,    # left_knee
    26: 14,    # right_knee
    27: 15,    # left_ankle
    28: 16     # right_ankle
}

# ==================================
# SKELETON DEFINITION (COCO format) – unchanged
# ==================================

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# ==================================
# HELPER FUNCTIONS
# ==================================

def ensure_frame_format(frame, width, height):
    """Ensure frame has correct format for video writer"""
    if frame is None or frame.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] != 3:
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            frame = frame[:, :, :3]
    if frame.shape[0] != height or frame.shape[1] != width:
        try:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        except:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
    return frame

def write_frame_with_retry(writer, frame, max_retries=3):
    for attempt in range(max_retries):
        try:
            writer.write(frame)
            return True
        except:
            if attempt < max_retries - 1:
                continue
            else:
                return False
    return False

# ==================================
# PATHS
# ==================================

VIDEO_PATH = "/Users/takneekmacmini/Documents/Reels Pipeline/IMG_2149.mp4"
OUTPUT_VIDEO = "cropped_output_final_2149_slow_mediapipe.mp4"
CSV_PATH = "person_boxes_2149_slow_mediapipe.csv"

# ==================================
# OPEN VIDEO
# ==================================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Cannot open video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total Frames:", total_frames)

fps_original = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output FPS for final playback speed
output_fps = fps_original * OUTPUT_SPEED_FACTOR
if output_fps < 1:
    output_fps = 1
    print(f"Warning: output FPS too low, set to 1")

print(f"Original FPS: {fps_original:.2f}")
print(f"Output FPS: {output_fps:.2f}")
print(f"Duplicate factor: {INPUT_DUPLICATE_FACTOR} (each frame written {INPUT_DUPLICATE_FACTOR} times)")

# ==================================
# 9:16 OUTPUT SIZE
# ==================================

BOX_HEIGHT = frame_height
BOX_WIDTH = int(BOX_HEIGHT * 9 / 16)
BOX_WIDTH = BOX_WIDTH if BOX_WIDTH % 2 == 0 else BOX_WIDTH + 1
BOX_HEIGHT = BOX_HEIGHT if BOX_HEIGHT % 2 == 0 else BOX_HEIGHT + 1

print(f"Output size - Width: {BOX_WIDTH}, Height: {BOX_HEIGHT}")

# ==================================
# OUTPUT VIDEO - Try codecs (for processed video)
# ==================================

codecs = [
    ('mp4v', '.mp4'),
    ('XVID', '.avi'),
    ('MJPG', '.avi'),
    ('X264', '.mp4'),
    ('H264', '.mp4')
]

writer = None
selected_codec = None

for codec, ext in codecs:
    try:
        test_output = f"test_output{ext}"
        test_writer = cv2.VideoWriter(
            test_output,
            cv2.VideoWriter_fourcc(*codec),
            output_fps,
            (BOX_WIDTH, BOX_HEIGHT)
        )
        if test_writer.isOpened():
            test_writer.release()
            if os.path.exists(test_output):
                os.remove(test_output)
            writer = cv2.VideoWriter(
                OUTPUT_VIDEO,
                cv2.VideoWriter_fourcc(*codec),
                output_fps,
                (BOX_WIDTH, BOX_HEIGHT)
            )
            if writer.isOpened():
                selected_codec = codec
                print(f"Using codec: {codec}")
                break
    except:
        continue

if writer is None or not writer.isOpened():
    print("Cannot create video writer with any codec")
    cap.release()
    exit()

# ==================================
# RAW SLOWED VIDEO WRITER (optional)
# ==================================

raw_writer = None
if SAVE_RAW_SLOWED_VIDEO:
    raw_dup_factor = 2
    raw_fps = fps_original
    raw_out_w = frame_width
    raw_out_h = frame_height
    for codec, ext in codecs:
        try:
            test_output = f"test_raw{ext}"
            test_writer = cv2.VideoWriter(
                test_output,
                cv2.VideoWriter_fourcc(*codec),
                raw_fps,
                (raw_out_w, raw_out_h)
            )
            if test_writer.isOpened():
                test_writer.release()
                if os.path.exists(test_output):
                    os.remove(test_output)
                raw_writer = cv2.VideoWriter(
                    RAW_SLOWED_OUTPUT,
                    cv2.VideoWriter_fourcc(*codec),
                    raw_fps,
                    (raw_out_w, raw_out_h)
                )
                if raw_writer.isOpened():
                    print(f"Raw slowed video will be saved to {RAW_SLOWED_OUTPUT} with FPS {raw_fps}")
                    break
        except:
            continue
    if raw_writer is None or not raw_writer.isOpened():
        print("Warning: Could not create raw slowed video writer. Skipping raw output.")
        SAVE_RAW_SLOWED_VIDEO = False

# ==================================
# CSV
# ==================================

csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_no",
    "top_left_x", "top_left_y",
    "top_right_x", "top_right_y",
    "bottom_right_x", "bottom_right_y",
    "bottom_left_x", "bottom_left_y"
])

# ==================================
# PROCESS VIDEO
# ==================================

frame_no = 0
output_frame_count = 0
error_frames = 0
max_errors = 50
previous_frame = None
duplicate_count = 0

while True:

    if duplicate_count == 0:
        ret, frame = cap.read()
        if not ret:
            print("Failed reading frame")
            break
        previous_frame = frame.copy()
        duplicate_count = 1
    else:
        frame = previous_frame.copy()
        duplicate_count = 0

    if SAVE_RAW_SLOWED_VIDEO and raw_writer is not None:
        raw_writer.write(frame)

    frame_no += 1

    # ---------- MediaPipe Pose detection ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    best_person = None
    best_keypoints = None

    if results.pose_landmarks:
        # Get landmarks and visibility
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Build COCO-like keypoints array (17 x 2)
        coco_kpts = np.full((17, 2), -1.0, dtype=np.float32)
        for mp_idx, coco_idx in MP_TO_COCO.items():
            if mp_idx < len(landmarks):
                lm = landmarks[mp_idx]
                if lm.visibility > 0.5:  # confidence threshold
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    coco_kpts[coco_idx] = [x, y]

        # Compute bounding box from visible keypoints
        visible = coco_kpts[coco_kpts[:, 0] >= 0]
        if len(visible) > 0:
            x_min = int(np.min(visible[:, 0]))
            x_max = int(np.max(visible[:, 0]))
            y_min = int(np.min(visible[:, 1]))
            y_max = int(np.max(visible[:, 1]))
            # Expand box by 20% to include some context
            pad_x = int((x_max - x_min) * 0.2)
            pad_y = int((y_max - y_min) * 0.2)
            x1 = max(0, x_min - pad_x)
            y1 = max(0, y_min - pad_y)
            x2 = min(frame_width, x_max + pad_x)
            y2 = min(frame_height, y_max + pad_y)
            best_person = (x1, y1, x2, y2)
            best_keypoints = coco_kpts

    # ---------- Prepare cropped frame ----------
    if best_person is not None:

        x1, y1, x2, y2 = best_person

        current_cx = (x1 + x2) / 2
        if smooth_cx is None:
            smooth_cx = current_cx
        else:
            smooth_cx = ALPHA * current_cx + (1 - ALPHA) * smooth_cx
        cx = int(smooth_cx)

        crop_x1 = int(cx - BOX_WIDTH / 2)
        crop_x2 = crop_x1 + BOX_WIDTH

        if crop_x1 < 0:
            crop_x1 = 0
            crop_x2 = BOX_WIDTH
        if crop_x2 > frame_width:
            crop_x2 = frame_width
            crop_x1 = frame_width - BOX_WIDTH

        crop_x1 = max(0, crop_x1)
        crop_x2 = min(frame_width, crop_x2)

        crop_y1 = 0
        crop_y2 = frame_height

        # Write to CSV (only once per unique frame)
        csv_writer.writerow([
            frame_no,
            crop_x1, crop_y1,
            crop_x2, crop_y1,
            crop_x2, crop_y2,
            crop_x1, crop_y2
        ])

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        if cropped.shape[0] != BOX_HEIGHT or cropped.shape[1] != BOX_WIDTH:
            try:
                cropped = cv2.resize(cropped, (BOX_WIDTH, BOX_HEIGHT), interpolation=cv2.INTER_LINEAR)
            except:
                cropped = np.zeros((BOX_HEIGHT, BOX_WIDTH, 3), dtype=np.uint8)

        # ---------- Draw skeleton ----------
        if best_keypoints is not None and cropped.size > 0:
            try:
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                if crop_w > 0 and crop_h > 0:
                    scale_x = BOX_WIDTH / crop_w
                    scale_y = BOX_HEIGHT / crop_h
                else:
                    scale_x = scale_y = 1.0

                LINE_COLOR = (255, 255, 255)  # White
                JOINT_COLOR = (32, 83, 249)   # #F95320 in BGR

                # Draw lines first
                for (i, j) in SKELETON:
                    if i < len(best_keypoints) and j < len(best_keypoints):
                        pt1 = best_keypoints[i]
                        pt2 = best_keypoints[j]
                        if (crop_x1 <= pt1[0] < crop_x2 and crop_y1 <= pt1[1] < crop_y2 and
                            crop_x1 <= pt2[0] < crop_x2 and crop_y1 <= pt2[1] < crop_y2):
                            lx1 = int((pt1[0] - crop_x1) * scale_x)
                            ly1 = int((pt1[1] - crop_y1) * scale_y)
                            lx2 = int((pt2[0] - crop_x1) * scale_x)
                            ly2 = int((pt2[1] - crop_y1) * scale_y)
                            lx1 = max(0, min(lx1, BOX_WIDTH-1))
                            ly1 = max(0, min(ly1, BOX_HEIGHT-1))
                            lx2 = max(0, min(lx2, BOX_WIDTH-1))
                            ly2 = max(0, min(ly2, BOX_HEIGHT-1))
                            cv2.line(cropped, (lx1, ly1), (lx2, ly2), LINE_COLOR, 8)

                # Draw joints on top (skip head)
                for idx, (x, y) in enumerate(best_keypoints):
                    if idx in [0, 1, 2, 3, 4]:
                        continue
                    if x < 0 or y < 0:   # invisible
                        continue
                    if crop_x1 <= x < crop_x2 and crop_y1 <= y < crop_y2:
                        lx = int((x - crop_x1) * scale_x)
                        ly = int((y - crop_y1) * scale_y)
                        lx = max(0, min(lx, BOX_WIDTH-1))
                        ly = max(0, min(ly, BOX_HEIGHT-1))
                        cv2.circle(cropped, (lx, ly), 10, JOINT_COLOR, -1)

            except Exception as e:
                print(f"Frame {frame_no}: Skeleton draw error - {e}")

        cropped = ensure_frame_format(cropped, BOX_WIDTH, BOX_HEIGHT)

        # ---------- Write frame N times for slowdown ----------
        success = True
        for _ in range(INPUT_DUPLICATE_FACTOR):
            if not write_frame_with_retry(writer, cropped):
                success = False
                break
        if not success:
            error_frames += 1
            print(f"Frame {frame_no}: Write failed, using blank frame")
            blank = np.zeros((BOX_HEIGHT, BOX_WIDTH, 3), dtype=np.uint8)
            for _ in range(INPUT_DUPLICATE_FACTOR):
                writer.write(blank)
            if error_frames > max_errors:
                print(f"Too many errors ({error_frames}), stopping...")
                break

        output_frame_count += INPUT_DUPLICATE_FACTOR

        try:
            display = cv2.resize(cropped, (360, 640))
            cv2.imshow("9:16 Crop with Skeleton", display)
        except:
            pass

    else:
        # No person – blank frame
        blank = np.zeros((BOX_HEIGHT, BOX_WIDTH, 3), dtype=np.uint8)
        for _ in range(INPUT_DUPLICATE_FACTOR):
            writer.write(blank)
        output_frame_count += INPUT_DUPLICATE_FACTOR
        try:
            display = cv2.resize(blank, (360, 640))
            cv2.imshow("9:16 Crop with Skeleton", display)
        except:
            pass

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# ==================================
# CLEANUP
# ==================================

cap.release()
writer.release()
if SAVE_RAW_SLOWED_VIDEO and raw_writer is not None:
    raw_writer.release()
csv_file.close()
cv2.destroyAllWindows()

print("--------------------------------")
print("Finished")
print("--------------------------------")
print("CSV Saved :", CSV_PATH)
print("Video Saved :", OUTPUT_VIDEO)
if SAVE_RAW_SLOWED_VIDEO and raw_writer is not None:
    print("Raw slowed video saved :", RAW_SLOWED_OUTPUT)
print(f"Total original frames: {frame_no}")
print(f"Total output frames written (processed): {output_frame_count}")
print(f"Output FPS: {output_fps:.2f}")
print(f"Total slowdown factor (processed): {INPUT_DUPLICATE_FACTOR / OUTPUT_SPEED_FACTOR:.2f}x")
print(f"Error frames: {error_frames}")
if selected_codec:
    print(f"Codec used: {selected_codec}")