import time
# =========================
# PID & Fuzzy-PID
# =========================
# Base gains 
Kp_ang_base, Ki_ang_base, Kd_ang_base = 0.83, 0.021, 0.012
Kp_spd_base, Ki_spd_base, Kd_spd_base = 0.60, 0.030, 0.005

# Trạng thái PID
previous_error_ang, integral_ang, previous_t_ang, d_filt_ang = 0.0, 0.0, None, 0.0
previous_error_spd, integral_spd, previous_t_spd, d_filt_spd = 0.0, 0.0, None, 0.0

def tri(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b:  return (x - a) / max(b - a, 1e-9)
    return (c - x) / max(c - b, 1e-9)

def fuzzy_scale(abs_e, abs_de):
    # Membership |e|
    eS = tri(abs_e, 0, 0, 5) + tri(abs_e, 0, 2.5, 5) * 0  
    eM = tri(abs_e, 3, 10, 18)
    eL = tri(abs_e, 12, 25, 45)
    # Membership |de|
    dS = tri(abs_de, 0, 0, 10)
    dM = tri(abs_de, 5, 20, 40)
    dL = tri(abs_de, 25, 60, 120)

    # Luật: 
    # 1) |e| lớn → tăng Kp (ưu tiên khi de nhỏ-trung bình)
    kp = 1.0 + 0.8 * max(min(eL, dS), min(eL, dM))
    # 2) |de| lớn → tăng Kd; đồng thời hãm Kp nhẹ
    kd = 0.6 + 0.8 * max(dL, min(eM, dM)) 
    kp -= 0.2 * dL
    kp = max(0.6, min(kp, 1.8))
    # 3) |e| nhỏ & |de| nhỏ → tăng Ki để xoá sai lệch nhỏ
    ki = 0.6 + 1.0 * min(max(eS, 0.4 * eM), dS)
    ki = max(0.5, min(ki, 1.6))
    # Giới hạn Kd đúng 
    kd = max(0.5, min(kd, 1.8))
    return kp, ki, kd

def pid_control_ang(target_angle, current_angle):
    global previous_error_ang, integral_ang, previous_t_ang, d_filt_ang, reset_time
    now = time.perf_counter()
    dt = 0.02 if previous_t_ang is None else max(1e-3, now - previous_t_ang)

    error = target_angle - current_angle
    derivative = (error - previous_error_ang) / dt

    sKp, sKi, sKd = fuzzy_scale(abs(error), abs(derivative))
    Kp_ang = Kp_ang_base * sKp
    Ki_ang = Ki_ang_base * sKi
    Kd_ang = Kd_ang_base * sKd

    # P
    P = Kp_ang * error
    # I (anti-windup)
    integral_ang += error * dt
    # clamping
    i_cap = 45.0 / max(Ki_ang, 1e-6)
    integral_ang = max(min(integral_ang, i_cap), -i_cap)
    I = Ki_ang * integral_ang
    # D (lọc IIR)
    d_filt_ang = 0.8 * d_filt_ang + 0.2 * derivative
    D = Kd_ang * d_filt_ang

    previous_error_ang = error
    previous_t_ang = now

    u = P + I + D
    return max(min(u, 45.0), -45.0)

def pid_control_spd(target_speed, current_speed):

    global previous_error_spd, integral_spd, previous_t_spd, d_filt_spd, reset_time

    now = time.perf_counter()
    dt = 0.02 if previous_t_spd is None else max(1e-3, now - previous_t_spd)

    error = target_speed - current_speed

    Kp_spd, Ki_spd, Kd_spd = Kp_spd_base, Ki_spd_base, Kd_spd_base

    # P
    P = Kp_spd * error
    # I (anti-windup)
    integral_spd += error * dt
    i_cap = 55.0 / max(Ki_spd, 1e-6)
    integral_spd = max(min(integral_spd, i_cap), -i_cap)
    I = Ki_spd * integral_spd
    # D (lọc IIR)
    derivative = (error - previous_error_spd) / dt
    d_filt_spd = 0.8 * d_filt_spd + 0.2 * derivative
    D = Kd_spd * d_filt_spd

    previous_error_spd = error
    previous_t_spd = now

    u = P + I + D
    return max(min(u, 55), 18)   
# =========================
# reset PID
# =========================
def reset_pid():
    global previous_error_ang, integral_ang, previous_t_ang, d_filt_ang
    global previous_error_spd, integral_spd, previous_t_spd, d_filt_spd
    previous_error_ang, integral_ang, previous_t_ang, d_filt_ang = 0.0, 0.0, None, 0.0
    previous_error_spd, integral_spd, previous_t_spd, d_filt_spd = 0.0, 0.0, None, 0.0
# =========================
