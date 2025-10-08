import cv2
import numpy as np
import math

DEBUG_VIEW = False 

BIAS_RIGHT = 0.715
CHECKPOINT_BASE = 112
previous_maxx, previous_minx = 0, 0

def showImage(gray_show, center_row, CHECKPOINT):
    h, w = gray_show.shape
    gray_show = cv2.line(gray_show, (center_row, CHECKPOINT), (int(w/2), h-1), 90, 2)
    gray_show = cv2.line(gray_show, (int(w/2), CHECKPOINT), (int(w/2), h-1), 90, 2)
    gray_show = cv2.line(gray_show, (int(w/2), CHECKPOINT), (center_row, CHECKPOINT), 90, 2)
    cv2.circle(gray_show, (center_row, CHECKPOINT), 5, (0, 0, 255), -1)
    cv2.imshow('Lane Tracking', gray_show)

# =========================
# Trích xuất góc từ mask
# =========================
def Ang_error(gray, c_spd):

    global previous_maxx, previous_minx

    h, w = gray.shape

    CHECKPOINT = int(CHECKPOINT_BASE + (40 - c_spd) * 0.8)
    CHECKPOINT = max(80, min(CHECKPOINT, h - 2)) 

    line_row = gray[CHECKPOINT, :]
    line = np.where(line_row == 255)[0]

    # fallback khi mất biên
    try:
        min_x = line[0]
    except Exception:
        min_x = previous_minx
    try:
        max_x = line[-1]
    except Exception:
        max_x = previous_maxx

    previous_minx = min_x
    previous_maxx = max_x

    center_row = int(min_x + BIAS_RIGHT * (max_x - min_x))

    if DEBUG_VIEW:
        showImage(gray.copy(), center_row, CHECKPOINT)

    # Ngã tư
    x0, y0 = int(w / 2), h
    x1, y1 = center_row, CHECKPOINT
    value = (x1 - x0) / max((y0 - y1), 1e-6)
    angle = math.degrees(math.atan(value)) / 2.0
    if (max_x - min_x) >= (200 + int((40 - c_spd)) * 0.7):
        angle = -0.3 * angle
    if len(line) < 2:
        return 0.0
    return angle


# =========================
# filter noise angle before PID
# =========================
angle_ema = 0.6
def filtered_angle(raw_angle, angle_alpha=0.25):
    global angle_ema
    angle_ema = (1 - angle_alpha) * angle_ema + angle_alpha * raw_angle
    return angle_ema

# =========================
# Scheduler speed according to cornering angle  
# =========================
def spd_cal(base_spd, angle, ang_max=25, spd_min=17):
    scale = max(0.0, 1.0 - (abs(angle) / ang_max))  
    spd = base_spd * (0.6 + 0.4 * scale)           
    return max(spd, spd_min)
