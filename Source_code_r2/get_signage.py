import time 

PRIORITY = {t: p for t, p in {
    "NO_LEFT":   2, 
    "NO_RIGHT":  2,
    "TURN_LEFT": 1,
    "TURN_RIGHT": 1,
    "STRAIGHT":  0,
}.items()}

CLASS_TO_TYPE = {
    "go_straight": "STRAIGHT",
    "right_turn": "TURN_RIGHT",
    "left_turn": "TURN_LEFT",
    "do_not_right_t": "NO_RIGHT",
    "do_not_left_t": "NO_LEFT",
    
    2: "STRAIGHT",
    4: "TURN_RIGHT",
    3: "TURN_LEFT",
    1: "NO_RIGHT",
    0: "NO_LEFT",
}

from collections import defaultdict, deque
SEEN_COUNT     = defaultdict(int)
LAST_TYPE      = None
MISSING_FRAMES = 0
MISSING_GRACE  = 3
DECAY_STEP     = 2
STABLE_COUNT   = 5
# Temporal majority vote over last K frames
from collections import deque
VOTE_WIN_SIZE   = 7     
VOTE_MIN_HITS   = 4      
VOTE_MIN_RATIO  = 0.57   
_vote_window = deque(maxlen=VOTE_WIN_SIZE)

# Cooldown n
COOLDOWN_S = {t: 0.1 for t in ["TURN_LEFT", "TURN_RIGHT", "NO_LEFT", "NO_RIGHT"]}
COOLDOWN_S["STRAIGHT"] = 0.05
last_emit_t = {k: 0.0 for k in COOLDOWN_S}

def tuple_box(det):
    # Nếu YOLO trả (x1, y1, x2, y2, conf, cls)
    if len(det) >= 6:
        x1, y1, x2, y2, conf, cls = det[:6]
    else:
        raise ValueError(f"Unexpected detection format: {det}")
    return int(x1), int(y1), int(x2), int(y2), float(conf), cls

def estimate_distance(box, frame_h, cfg: dict):
    x1, y1, x2, y2, _, _ = box

    # get bottom bbox
    y_bottom = min(frame_h - 1, int(max(y1, y2)))

    # distance form bottom frame to bottom bbox
    dy = max(cfg.get("y_ref_from_bottom", 10), (frame_h - 1) - y_bottom)

    k = float(cfg.get("k_m_per_pix_y", 1.0))

    return max(0.01, dy * k)

def _decay_seen():
    for k in list(SEEN_COUNT.keys()):
        SEEN_COUNT[k] = max(0, SEEN_COUNT[k] - DECAY_STEP)
        if SEEN_COUNT[k] == 0:
            del SEEN_COUNT[k]

cam_cfg = {
    "k_m_per_pix_y": 0.02,    
    "y_ref_from_bottom": 10,   
}

def get_sign_event(bboxes, gray_show, conf_thresh=0.76, cam_cfg=cam_cfg):
    global LAST_TYPE, MISSING_FRAMES, _vote_window

    height_frame = gray_show.shape[0]

    # Handle empty frame with grace window
    if not bboxes:
        MISSING_FRAMES += 1
        if MISSING_FRAMES <= MISSING_GRACE:
            _decay_seen()
            return None
        LAST_TYPE = None
        SEEN_COUNT.clear()
        return None
    else:
        MISSING_FRAMES = 0

    # ---- per-frame pick best ----
    picked = None
    picked_types = None
    picked_score = 0.0
    picked_yb = 0.0
    best_key = (-1, 0.0, -1.0)  # (priority, conf, y_bottom_norm)
    best_box_per_type = {}

    for det in bboxes:
        x1, y1, x2, y2, score, cls_sign = tuple_box(det)
        if score < conf_thresh:
            continue
        typ = CLASS_TO_TYPE.get(cls_sign)
        if not typ:
            continue

        pr = PRIORITY.get(typ, 0)
        y_bottom = min(height_frame - 1, int(max(y1, y2)))
        yb_norm = y_bottom / float(max(1, height_frame - 1))
        key = (pr, float(score), yb_norm)

        if key > best_key:
            best_key = key
            picked = (x1, y1, x2, y2, float(score), cls_sign)
            picked_types, picked_score, picked_yb = typ, float(score), yb_norm

        cur = best_box_per_type.get(typ)
        if (cur is None) or (float(score), yb_norm) > (cur[0], cur[1]):
            best_box_per_type[typ] = (float(score), yb_norm, (x1, y1, x2, y2, float(score), cls_sign))

    if picked is None:
        MISSING_FRAMES += 1
        if MISSING_FRAMES <= MISSING_GRACE:
            _decay_seen()
            return None
        LAST_TYPE = None
        SEEN_COUNT.clear()
        return None

    _vote_window.append(picked_types)
    counts = {}
    for t in _vote_window:
        counts[t] = counts.get(t, 0) + 1

    def majority_key(t):
        return (counts[t], PRIORITY.get(t, 0))

    winner = max(counts.keys(), key=majority_key)
    win_hits = counts[winner]
    win_ratio = win_hits / len(_vote_window)
    majority_ok = (win_hits >= VOTE_MIN_HITS) and (win_ratio >= VOTE_MIN_RATIO)

    if not majority_ok:
        return None

    if winner in best_box_per_type:
        picked_types = winner
        picked_score, picked_yb, picked = best_box_per_type[winner]
    else:
        return None

    if picked_types == LAST_TYPE:
        SEEN_COUNT[picked_types] += 1
    else:
        SEEN_COUNT[picked_types] = max(1, SEEN_COUNT.get(picked_types, 0) + 1)
        if LAST_TYPE and LAST_TYPE in SEEN_COUNT:
            SEEN_COUNT[LAST_TYPE] = max(0, SEEN_COUNT[LAST_TYPE] - 1)
        LAST_TYPE = picked_types

    if SEEN_COUNT[picked_types] < STABLE_COUNT:
        return None

    now = time.perf_counter()
    last_t = last_emit_t.get(picked_types, 0.0)
    allow_emit = (now - last_t) >= COOLDOWN_S.get(picked_types, 0.2)
    if not allow_emit and picked_yb > 0.8:
        allow_emit = True
    if not allow_emit:
        return None

    dist = estimate_distance(picked, height_frame, cam_cfg)
    last_emit_t[picked_types] = now
    print(f"✅ Stable sign: {picked_types} ({SEEN_COUNT[picked_types]} frames) "
          f"dist={dist:.2f}m conf={picked_score:.2f}")
    return {"type": picked_types, "dist": dist, "conf": picked_score}