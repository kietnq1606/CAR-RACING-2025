import time
import math


class signFSM:
    def __init__(self):
        self.turn_required = None
        self.turn_forbidden = set()

        self.go_straight = 0
        self.last_seen_type = None
        self.last_seen_time = 0.0
        self.last_seen_dist = 0.0  # distance when last seen (m)
    
        self.hold_time_s_no = 2.8
        self.hold_time_s = 4.5   # giữ 2 giây nếu không có biển mới
        self.hold_dist_m = 2.5    # hoặc 8m 
    def on_sign(self, event):
        # event = {"t
        #type": str, "dist": float, "conf": float}
        if not event:
            return

        t = event.get("type")
        dis = event.get("dist")
        # Update last seen
        self.last_seen_type = t
        self.last_seen_time = time.perf_counter()
        self.last_seen_dist = event.get("dist", 0.0)

        if t == "TURN_LEFT":
            self.turn_required = "LEFT"
            self.turn_forbidden.discard("LEFT")
            self.turn_forbidden.discard("RIGHT")
            self.go_straight = 0

        elif t == "TURN_RIGHT":
            self.turn_required = "RIGHT"
            self.turn_forbidden.discard("LEFT")
            self.turn_forbidden.discard("RIGHT")
            self.go_straight = 0


        elif t == "STRAIGHT":
            self.turn_required = None
            self.turn_forbidden.clear()
            self.go_straight = 1

        elif t == "NO_LEFT":
            self.turn_forbidden.add("LEFT")
            if self.turn_required == "LEFT":
                self.turn_required = None
            self.go_straight = 0

        elif t == "NO_RIGHT":
            self.turn_forbidden.add("RIGHT")
            if self.turn_required == "RIGHT":
                self.turn_required = None
            self.go_straight = 0

    def _expired(self, current_dist=None):
        dt = time.perf_counter() - self.last_seen_time
        
        if dt > self.hold_time_s:
            return True
        if current_dist is not None and abs(current_dist - self.last_seen_dist) > self.hold_dist_m:
            return True
        return False
    def _expired_no_turn(self, current_dist=None):
        dt = time.perf_counter() - self.last_seen_time
        if dt > self.hold_time_s_no:
            return True
        if current_dist is not None and abs(current_dist - self.last_seen_dist) > self.hold_dist_m:
            return True
        return False
    
    def process_sign(self, current_dist=None):
        # reset signage 
        if self.last_seen_type not in ('NO_RIGHT', 'NO_LEFT'):
            if self.last_seen_type and self._expired(current_dist):
                self.turn_required = None
                self.turn_forbidden.clear()
                self.last_seen_type = None
        elif self.last_seen_type in ('NO_RIGHT', 'NO_LEFT'):
            if self.last_seen_type and self._expired_no_turn(current_dist):
                self.turn_required = None
                self.turn_forbidden.clear()
                self.last_seen_type = None

        # handle no turn signs
        if "LEFT" in self.turn_forbidden and "RIGHT" not in self.turn_forbidden:
            return "NO_LEFT"
        elif "RIGHT" in self.turn_forbidden and "LEFT" not in self.turn_forbidden:
            return "NO_RIGHT"
        elif "LEFT" in self.turn_forbidden and "RIGHT" in self.turn_forbidden:
            return "NO_TURN"
        # handle when required turn left/right 
        elif self.turn_required in ("LEFT", "RIGHT"):
            lateral = "TURN_LEFT" if self.turn_required == "LEFT" else "TURN_RIGHT"
            return lateral
        elif self.go_straight:
            return 'STRAIGHT'
        return 'NORMAL'
