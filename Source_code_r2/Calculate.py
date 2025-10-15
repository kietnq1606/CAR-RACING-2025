import cv2
import numpy as np
import math
import time

NgaTu = 0
DEBUG_VIEW = True

BIAS_RIGHT = 0.5
CHECKPOINT_BASE = 108
previous_maxx, previous_minx = 0, 0
previous_angle = 0.0
last_dis = 0.0
def showImage(gray_show, center_row, CHECKPOINT):
    h, w = gray_show.shape
    gray_show = cv2.line(gray_show, (center_row, CHECKPOINT), (int(w/2), h-1), 90, 2)
    gray_show = cv2.line(gray_show, (int(w/2), CHECKPOINT), (int(w/2), h-1), 90, 2)
    gray_show = cv2.line(gray_show, (int(w/2), CHECKPOINT), (center_row, CHECKPOINT), 90, 2)
    cv2.circle(gray_show, (center_row, CHECKPOINT), 5, (0, 0, 255), -1)
    cv2.imshow('Lane Tracking', gray_show)

def Ang_error(gray, c_spd, sign=None, dis = 0.0):

    global previous_minx, previous_maxx, previous_angle, NgaTu, last_dis
    angle = previous_angle
    h, w = gray.shape
    
    CHECKPOINT = int(CHECKPOINT_BASE + (40 - c_spd) * 1)
    

    line_row = gray[CHECKPOINT, :]
    line = np.flatnonzero(line_row == 255)

    if len(line) >= 2:
        min_x, max_x = line[0], line[-1]
    else:
        min_x, max_x = previous_minx, previous_maxx
    previous_minx, previous_maxx = min_x, max_x
    if sign in ("TURN_LEFT"):
        if last_dis > 3:
            #min_x -= 12
            CHECKPOINT += int((40 + c_spd*0.8))
        if dis:
            last_dis = dis 
    if sign in ( "TRURN_RIGHT"):
        if last_dis > 3:
            #max_x += 12
            CHECKPOINT += int((40 + c_spd*0.8))
        if dis:
            last_dis = dis 
    if sign in ( "NO_LEFT"):
        if last_dis > 3:
            max_x += 12
            CHECKPOINT += 25
        if dis:
            last_dis = dis 
    if sign in ( "NO_RIGHT"):
        if last_dis > 3:
            max_x -= 12
            CHECKPOINT += 25
        if dis:
            last_dis = dis 
    # ======= process when meet signage =======
    print(f"min: {min_x} \t max: {max_x}")
    center_row = int(min_x + BIAS_RIGHT * (max_x - min_x))
    if sign == 'STRAIGHT' and (max_x - min_x) >= (190 + int((40 - c_spd)) * 0.7):
            return 0.0
    #  ====== process when meet junction =======
    if (max_x - min_x) >= (242 + int((40 - c_spd)) * 0.7):
        NgaTu = 1
        
        if sign == "TURN_LEFT":
            
            if last_dis > 3.0:
                return -50
            else: 
                return -40
        elif sign == "TURN_RIGHT":
            if last_dis > 3.0:
                return 50
            else:
                return 40
    
    center_row = int(min_x + BIAS_RIGHT * (max_x - min_x))
    x0, y0 = w // 2, h
    x1, y1 = center_row, CHECKPOINT
    value = (x1 - x0) / max((y0 - y1), 1e-6)
    angle = math.degrees(math.atan(value))
    if sign == "NO_LEFT":
        if last_dis > 3.2:
            angle += 17
    elif sign == "NO_RIGHT":
        if last_dis > 3.2:
            angle -= 17
    else:
        NgaTu = 0 
        x0, y0 = w // 2, h
        x1, y1 = center_row, CHECKPOINT
        value = (x1 - x0) / max((y0 - y1), 1e-6)
        angle = math.degrees(math.atan(value))
        
    # ========= handling when border is lost ========
    if len(line) < 2:
        if len(line) == 1:
            if line[0] > w // 2:
                center_row = int(line[0] - w * 0.10)
            else:
                center_row = int(line[0] + w * 0.10)
        else:
            angle = previous_angle * 0.7  # giảm dần góc cũ để tránh lệch khi mất line
            previous_angle = angle
            return angle
        
    if DEBUG_VIEW:
        showImage(gray.copy(), center_row, CHECKPOINT)
    # ======= calculate angle =======
    print(f"dis = {dis}")
    previous_angle = angle
    return angle

# =========================
# filter noise angle before PID
# =========================
angle_ema = 0.6
def filtered_angle(raw_angle, angle_alpha=0.6):
    global angle_ema
    angle_ema = (1 - angle_alpha) * angle_ema + angle_alpha * raw_angle
    return angle_ema

# =========================
# Scheduler speed according to cornering angle  
# =========================
def spd_cal(base_spd, angle, ang_max=25, spd_min=17, sign = None):
        # giảm tốc khi cua hoặc có biển cấm
    global NgaTu
    spd = 0
    if not NgaTu:
        if sign in ("TURN_LEFT", "TURN_RIGHT"):
            base_spd *= -0.5
        elif sign in ("NO_LEFT", "NO_RIGHT"):
            base_spd *= -0.5
        elif sign == "STRAIGHT" or sign is None: 
            base_spd = base_spd
    
    scale = max(0.0, 1.0 - (abs(angle) / ang_max))  
    spd = base_spd * (0.6 + 0.4 * scale)           
    return max(spd, spd_min)
