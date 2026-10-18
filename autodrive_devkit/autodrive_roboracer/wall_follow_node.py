# Ada Onder's RoboRacer Technical Challenge Code
################################################################################
#
# Run:  python3 wall_follow_node_v12.py         (SAFE profile)
#       python3 wall_follow_node_v12.py fast    (FAST profile)
#       python3 wall_follow_node_v12.py nowobble  (A/B: part P off)
################################################################################

import math
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32
from nav_msgs.msg import Odometry

# ----------------------------------------------------------------------------
# Topics
# ----------------------------------------------------------------------------
SCAN_TOPIC = "/autodrive/roboracer_1/lidar"
THROTTLE_TOPIC = "/autodrive/roboracer_1/throttle_command"
STEERING_TOPIC = "/autodrive/roboracer_1/steering_command"
SPEED_TOPIC = "/autodrive/roboracer_1/speed"   # (not published by this sim)
ODOM_TOPIC = "/autodrive/roboracer_1/odom"       # logged only
LAST_LAP_TOPIC = "/autodrive/roboracer_1/last_lap_time"
COLLISION_TOPIC = "/autodrive/roboracer_1/collision_count"

# ----------------------------------------------------------------------------
# Vehicle / simulator
# ----------------------------------------------------------------------------
MAX_STEER_RAD = 0.5236
STEER_SIGN = 1.0
WHEELBASE = 0.32
CAR_HALF_WIDTH = 0.15
THROTTLE_PER_MPS = 1.0 / 24.8
COAST_DECEL = 3.0
SPEED_EST_UP = 8.0
RANGE_CAP = 10.0
SELF_RANGE = 0.12

# ----------------------------------------------------------------------------
# Wall fitting
# ----------------------------------------------------------------------------
WALL_SECTOR_MIN_DEG = 20.0
WALL_SECTOR_MAX_DEG = 115.0
WALL_MAX_RANGE = 4.0
WALL_X_MIN = -0.6
WALL_X_MAX = 1.5
MIN_WALL_POINTS = 15
WALL_MAX_RESIDUAL = 0.15
WALL_MAX_HEADING_DEG = 60.0
DEFAULT_TRACK_WIDTH = 1.4
MAX_TRACK_WIDTH = 4.0

# ----------------------------------------------------------------------------
# Steering
# ----------------------------------------------------------------------------
K_HEADING = 0.9
K_CROSS = 1.1
K_SOFT = 1.0
K_DAMP = 0.03
STEER_TAU = 0.02

GAP_FOV_DEG = 100.0
LOOK_BASE = 0.9
LOOK_TIME = 0.22
LOOK_MARGIN = 0.08
GAP_SMOOTH_TAU = 0.03
K_GAP = 2.1                 
W_BASE = 0.15                
CORNER_TRIGGER_BASE = 1.3    
CORNER_TRIGGER_TIME = 0.30   
CORNER_FULL = 0.45           

SIDE_SAFE = 0.22
SIDE_X_MIN = -0.2
SIDE_X_MAX = 0.8
K_SAFE = 1.2

# ----------------------------------------------------------------------------
# Speed (m/s)
# ----------------------------------------------------------------------------
V_MAX = 3.9                 
V_MIN = 0.8
A_BRAKE_PLAN = 2.8
A_LAT = 4.4                  
STOP_MARGIN = 0.25
LATENCY = 0.10               # s, planned delay at the normal scan rate
NOMINAL_DT = 0.07            # s, normal scan period (~14 Hz measured)
SIDE_SLOW_DIST = 0.18
V_SIDE = 2.5
ACCEL_RATE = 6.0
CORRIDOR_MARGIN = 0.05
CORRIDOR_X_MIN = 0.25
PATH_MAX_SWEEP_DEG = 120.0   # how far around the arc we look for obstacles
PATH_ARCS = 4                # arcs tried between current and commanded steering
KAPPA_MAX = math.tan(0.5236) / 0.32   # tightest arc the car can steer

WATCHDOG_S = 0.5
LOG_EVERY = 20
STATS_EVERY = 400            # scans between limiter-share summaries

FAST_MODE = True
FAST_OVERRIDES = dict(
    V_MAX=4.3,
    A_LAT=5.0,
    A_BRAKE_PLAN=3.3,
    STOP_MARGIN=0.15,
    LATENCY=0.08,
    V_SIDE=3.0,
)
PROFILE = "SAFE"

USE_ODOM_PLAN = True         # L. planning speed = max(estimate, real)
K_BOOST = 0.0                # M. off in v11 (flag "boost" -> 0.5)
BOOST_MAX = 1.0              # m/s, most the command may exceed the target
K_BRAKE = 0.0                # M. off in v11 (flag "brake" -> 1.0)
LAG_DEADBAND = 0.2           # m/s, ignore smaller differences
ODOM_FRESH_S = 0.3           # s, odom older than this is not used

# v11
ANTI_WOBBLE = True           # P. speed-scaled heading gain + delay prediction
HEADING_V_REF = 2.0          # m/s, full K_HEADING up to this speed
ENTRY_LIMIT = True           # Q. corner-entry speed limit from wall ahead
ENTRY_MARGIN = 0.35          # m, kept between the turning arc and the wall
ENTRY_LEAD_T = 0.25          # s, real speed lags the command this much


def apply_profile(args):
    global PROFILE, USE_ODOM_PLAN, K_BOOST, K_BRAKE, ANTI_WOBBLE, ENTRY_LIMIT
    if FAST_MODE or "fast" in args:
        globals().update(FAST_OVERRIDES)
        PROFILE = "FAST"
    if "noplan" in args:
        USE_ODOM_PLAN = False
    if "boost" in args:
        K_BOOST = 0.5
    if "brake" in args:
        K_BRAKE = 1.0
    if "nowobble" in args:
        ANTI_WOBBLE = False
    if "noentry" in args:
        ENTRY_LIMIT = False
    if "nolead" in args:
        globals()["ENTRY_LEAD_T"] = 0.0


def fit_line(x, y):
    """Robust least-squares fit y = m*x + c. Returns (m, c, rms) or None."""
    if len(x) < MIN_WALL_POINTS:
        return None
    m = c = 0.0
    for _ in range(3):
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        res = np.abs(y - (m * x + c))
        keep = res < max(0.05, 2.5 * float(np.median(res)))
        if keep.sum() < MIN_WALL_POINTS or keep.all():
            break
        x, y = x[keep], y[keep]
    rms = float(np.sqrt(np.mean((y - (m * x + c)) ** 2)))
    return float(m), float(c), rms


def path_clearance(x, y, valid, kappa):
    """Free distance (m) along a constant-curvature arc of the car's width.

    kappa > 0 = turning left. Points whose distance from the arc is less than
    the car half width (+ margin) count as obstacles on the path; the result is
    the arc length to the nearest one.
    """
    hw = CAR_HALF_WIDTH + CORRIDOR_MARGIN
    if abs(kappa) < 1e-3:
        sel = valid & (x > CORRIDOR_X_MIN) & (np.abs(y) < hw)
        return float(np.min(x[sel])) if np.any(sel) else RANGE_CAP
    R = 1.0 / kappa                     # signed radius, centre at (0, R)
    dev = np.abs(np.hypot(x, y - R) - abs(R))
    swept = np.arctan2(x, math.copysign(1.0, R) * (R - y))   # 0 at the car
    s = abs(R) * swept
    sel = (valid & (dev < hw) & (swept > 0.0)
           & (swept < math.radians(PATH_MAX_SWEEP_DEG)) & (s > CORRIDOR_X_MIN))
    # Nothing on the inspected part of the arc: only that much is known free.
    seen = min(RANGE_CAP, abs(R) * math.radians(PATH_MAX_SWEEP_DEG))
    return float(min(np.min(s[sel]), seen)) if np.any(sel) else seen


class WallFollowRacer(Node):

    def __init__(self):
        super().__init__("roboracer_wall_follower")

        qos = QoSProfile(depth=1)
        self.pub_throttle = self.create_publisher(Float32, THROTTLE_TOPIC, qos)
        self.pub_steering = self.create_publisher(Float32, STEERING_TOPIC, qos)
        self.create_subscription(
            LaserScan, SCAN_TOPIC, self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Float32, SPEED_TOPIC, self.speed_callback, qos)
        self.create_timer(0.1, self.watchdog)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32, LAST_LAP_TOPIC, self.last_lap_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Int32, COLLISION_TOPIC, self.collision_callback,
                                 qos_profile_sensor_data)
        self.v_odom = None
        self.v_odom_time = 0.0
        self.a_odom = 0.0             # filtered real acceleration (m/s^2)
        self.a_max = 0.0              # STATS window: hardest real accel
        self.d_max = 0.0              # STATS window: hardest real braking
        self.err_up = []              # est - real while accelerating
        self.err_dn = []              # est - real while braking
        self.n_boost = 0
        self.n_brake = 0
        self.last_lap = None
        self.best_lap = None
        self.laps_logged = 0
        self.collisions = 0
        self.collisions_at_lap_start = 0

        self.n = 0
        self.track_width = DEFAULT_TRACK_WIDTH
        self.prev_e = 0.0
        self.e_dot = 0.0
        self.prev_t = None
        self.steer_rad = 0.0
        self.theta_gap = 0.0
        self.v_cmd = 0.0
        self.v_est = 0.0
        self.v_meas = None
        self.v_meas_time = 0.0
        self.last_scan_wall = time.monotonic()
        self.count = 0
        self.lim_counts = {}
        self.v_sum = 0.0
        self.min_body = RANGE_CAP    # closest wall beside the car (window)
        self.min_front = RANGE_CAP   # closest wall straight ahead (window)
        self.v_close = 0.0           # fastest speed with a wall < 1 m ahead
        self.dt_peak = NOMINAL_DT

    def speed_callback(self, msg):
        self.v_meas = abs(float(msg.data))
        self.v_meas_time = time.monotonic()

    def odom_callback(self, msg):
        lin = msg.twist.twist.linear
        v = math.hypot(float(lin.x), float(lin.y))
        now = time.monotonic()
        if self.v_odom is not None:
            dt = now - self.v_odom_time
            if 0.005 < dt < 0.3:
                self.a_odom += 0.3 * ((v - self.v_odom) / dt - self.a_odom)
                self.a_max = max(self.a_max, self.a_odom)
                self.d_max = max(self.d_max, -self.a_odom)
        self.v_odom = v
        self.v_odom_time = now

    def collision_callback(self, msg):
        self.collisions = int(msg.data)

    def last_lap_callback(self, msg):
        t = float(msg.data)
        if not math.isfinite(t) or t <= 0.0:
            return
        if self.last_lap is not None and abs(t - self.last_lap) < 1e-3:
            return
        self.last_lap = t
        self.laps_logged += 1
        if self.best_lap is None or t < self.best_lap:
            self.best_lap = t
        col = self.collisions - self.collisions_at_lap_start
        self.collisions_at_lap_start = self.collisions
        self.get_logger().info(
            f"LAP {self.laps_logged} done: {t:.2f} s (best {self.best_lap:.2f} s), "
            f"collisions this lap {col}")

    def watchdog(self):
        if time.monotonic() - self.last_scan_wall > WATCHDOG_S:
            self.v_cmd = 0.0
            self.publish(0.0, 0.0)

    def _init_geometry(self, scan, n):
        self.n = n
        a = scan.angle_min + np.arange(n) * scan.angle_increment
        self.angles = a
        self.cos = np.cos(a)
        self.sin = np.sin(a)
        lo, hi = math.radians(WALL_SECTOR_MIN_DEG), math.radians(WALL_SECTOR_MAX_DEG)
        self.left_mask = (a >= lo) & (a <= hi)
        self.right_mask = (a <= -lo) & (a >= -hi)
        self.gap_mask = np.abs(a) <= math.radians(GAP_FOV_DEG)
        self.gap_angles = a[self.gap_mask]

    def _wall(self, mask, valid, r, x, y, side):
        sel = mask & valid & (r < WALL_MAX_RANGE)
        xs, ys = x[sel], y[sel]
        keep = (xs > WALL_X_MIN) & (xs < WALL_X_MAX) & (side * ys > 0.05)
        fit = fit_line(xs[keep], ys[keep])
        if fit is None:
            return None
        m, c, rms = fit
        heading = math.atan(m)
        if rms > WALL_MAX_RESIDUAL or abs(heading) > math.radians(WALL_MAX_HEADING_DEG):
            return None
        dist = side * c / math.sqrt(1.0 + m * m)
        if dist <= 0.05:
            return None
        return dist, heading

    def scan_callback(self, scan):
        self.last_scan_wall = time.monotonic()
        r = np.asarray(scan.ranges, dtype=np.float64)
        if len(r) != self.n:
            self._init_geometry(scan, len(r))

        valid = np.isfinite(r) & (r >= max(scan.range_min, SELF_RANGE)) & (r <= RANGE_CAP)
        r_open = np.where(np.isfinite(r), np.clip(r, 0.0, RANGE_CAP), RANGE_CAP)
        r_open = np.where(np.isfinite(r) & (r < SELF_RANGE), RANGE_CAP, r_open)
        rv = np.where(valid, r, RANGE_CAP)
        x = rv * self.cos
        y = rv * self.sin

        t = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        if t <= 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        dt = (t - self.prev_t) if self.prev_t is not None else 0.0
        self.prev_t = t
        dt_f = float(np.clip(dt, 0.01, 0.2))
        self.dt_peak = max(dt_f, self.dt_peak * 0.97)
        latency = max(LATENCY, LATENCY + (self.dt_peak - NOMINAL_DT))

        measured = (self.v_meas is not None
                    and time.monotonic() - self.v_meas_time < 0.5)
        if measured:
            self.v_est = self.v_meas
        elif self.v_cmd > self.v_est:
            self.v_est = min(self.v_cmd, self.v_est + SPEED_EST_UP * dt_f)
        else:
            self.v_est = max(self.v_cmd, self.v_est - COAST_DECEL * dt_f)
        odom_ok = (self.v_odom is not None
                   and time.monotonic() - self.v_odom_time < ODOM_FRESH_S)
        v = self.v_est
        if USE_ODOM_PLAN and odom_ok:
            # L. never plan with less than the real speed (matters when braking)
            v = max(v, self.v_odom)

        # 1. Walls
        left = self._wall(self.left_mask, valid, r, x, y, +1)
        right = self._wall(self.right_mask, valid, r, x, y, -1)

        have_wall = True
        if left and right:
            width = left[0] + right[0]
            if width < MAX_TRACK_WIDTH:
                self.track_width = 0.9 * self.track_width + 0.1 * width
            e = 0.5 * (left[0] - right[0])
            psi = 0.5 * (left[1] + right[1])
        elif left:
            e = left[0] - 0.5 * self.track_width
            psi = left[1]
        elif right:
            e = 0.5 * self.track_width - right[0]
            psi = right[1]
        else:
            e, psi, have_wall = 0.0, 0.0, False

        if dt > 1e-4:
            raw = (e - self.prev_e) / dt
            self.e_dot = 0.7 * self.e_dot + 0.3 * float(np.clip(raw, -3.0, 3.0))
        self.prev_e = e

        # 2. Stanley steering
        k_head = K_HEADING
        e_s, psi_s = e, psi
        if ANTI_WOBBLE:
            # P1. The car turns faster at speed (yaw rate ~ v * steer), so the
            # same heading gain overshoots more. Keep the loop speed constant.
            k_head = K_HEADING * min(1.0, HEADING_V_REF / max(v, 0.5))
            # P2. Steer for where the car will be when the command takes effect
            # (scan age + latency), using the real speed when available.
            v_real = self.v_odom if odom_ok else v
            t_p = latency
            yaw_rate = v_real * math.tan(self.steer_rad) / WHEELBASE
            psi_s = psi - yaw_rate * t_p
            e_s = e + v_real * math.sin(psi) * t_p
        delta_wall = (k_head * psi_s
                      + math.atan2(K_CROSS * e_s, v + K_SOFT)
                      + K_DAMP * self.e_dot)

        # 3. Lookahead-circle corner steering (straight-ahead "front" triggers it)
        front = path_clearance(x, y, valid, 0.0)

        Ld = LOOK_BASE + LOOK_TIME * v
        ro = r_open[self.gap_mask]
        free = (ro > Ld).astype(np.int8)
        k = int(math.ceil(math.atan2(CAR_HALF_WIDTH + LOOK_MARGIN, Ld)
                          / scan.angle_increment))
        padded = np.pad(free, k, mode="constant")
        free = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1).min(axis=1)
        edges = np.diff(np.concatenate(([0], free, [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
        if len(starts):
            centers = 0.5 * (self.gap_angles[starts] + self.gap_angles[ends])
            target = float(centers[int(np.argmin(np.abs(centers - self.theta_gap)))])
        else:
            target = float(self.gap_angles[int(np.argmax(ro))])
        a_g = dt_f / (GAP_SMOOTH_TAU + dt_f)
        self.theta_gap += a_g * (target - self.theta_gap)
        theta_gap = self.theta_gap
        delta_gap = math.atan2(2.0 * WHEELBASE * math.sin(theta_gap), Ld) * K_GAP

        trigger = CORNER_TRIGGER_BASE + CORNER_TRIGGER_TIME * v
        w = float(np.clip((trigger - front) / max(trigger - CORNER_FULL, 1e-3), 0.0, 1.0))
        w = max(w, W_BASE)
        if not have_wall:
            w = 1.0
        delta = (1.0 - w) * delta_wall + w * delta_gap

        # 4. Side safety push
        box = valid & (x > SIDE_X_MIN) & (x < SIDE_X_MAX)
        yl = y[box & (y > 0.0)]
        yr = -y[box & (y < 0.0)]
        cl = float(yl.min()) if yl.size else RANGE_CAP
        cr = float(yr.min()) if yr.size else RANGE_CAP
        delta += K_SAFE * (max(0.0, SIDE_SAFE - cr) - max(0.0, SIDE_SAFE - cl))

        delta = float(np.clip(delta, -MAX_STEER_RAD, MAX_STEER_RAD))
        a_s = dt_f / (STEER_TAU + dt_f)
        self.steer_rad += a_s * (delta - self.steer_rad)
        steering = float(np.clip(self.steer_rad / MAX_STEER_RAD, -1.0, 1.0)) * STEER_SIGN

        # 5. Speed. Stopping distance is measured along the arc we are steering on.
        A_LAT_STEER = A_LAT * 1.15 * 1.08 ** 2   # lateral grip the steer limit allows
        kappa_path = math.tan(self.steer_rad) / WHEELBASE
        kappa_cmd = math.tan(delta) / WHEELBASE
        # Only arcs the car can actually hold at its current speed count as an
        # escape; a tight arc into the corner is not one at full speed.
        kappa_ok = A_LAT_STEER / max(v, 0.5) ** 2
        cands = [0.0, kappa_path]
        cands += [float(k) for k in np.linspace(kappa_path, kappa_cmd, PATH_ARCS)
                  if abs(k) <= min(kappa_ok, KAPPA_MAX)]
        front_path = max(path_clearance(x, y, valid, k) for k in cands)
        reach = front_path - STOP_MARGIN - v * latency
        v_stop = math.sqrt(2.0 * A_BRAKE_PLAN * max(reach, 0.0)) * 1.10
        kappa_gap = abs(2.0 * math.sin(theta_gap) / Ld)
        v_corner = math.sqrt(A_LAT / max(kappa_gap, 1e-3))
        kappa_steer = abs(kappa_path)
        v_steer = math.sqrt((A_LAT * 1.15) / max(kappa_steer, 1e-3))
        v_side = V_SIDE if min(cl, cr) < SIDE_SLOW_DIST else V_MAX
        limits = dict(max=V_MAX, stop=v_stop * 1.08, corner=v_corner * 1.12,
                      steer=v_steer * 1.08, side=v_side)
        if ENTRY_LIMIT:
            # Q. Corner entry: with a wall d ahead (straight and along the
            # current arc), the car must be slow enough to turn on a radius
            # of about d - margin. Slows progressively before the corner
            # instead of at full lock with the wall 0.3 m ahead.
            d_ahead = max(front, path_clearance(x, y, valid, kappa_path))
            d_ahead -= v * ENTRY_LEAD_T   # R. reach entry speed in time
            r_turn = max(d_ahead - ENTRY_MARGIN, 1.0 / KAPPA_MAX)
            limits["entry"] = math.sqrt(A_LAT * 1.15 * r_turn) * 1.08
        lim = min(limits, key=limits.get)
        v_target = max(V_MIN, limits[lim])

        if v_target > self.v_cmd:
            self.v_cmd = min(v_target, self.v_cmd + ACCEL_RATE * dt_f)
        else:
            self.v_cmd = v_target

        # M. Compensate the lag between command and real speed.
        v_set = self.v_cmd
        if odom_ok:
            lag = self.v_cmd - self.v_odom
            if lag > LAG_DEADBAND and lim != "stop" and K_BOOST > 0.0:
                v_set = self.v_cmd + min(K_BOOST * lag, BOOST_MAX)
                self.n_boost += 1
            elif lag < -LAG_DEADBAND and K_BRAKE > 0.0:
                v_set = max(0.0, self.v_cmd + K_BRAKE * lag)
                self.n_brake += 1
            if self.v_est > self.v_odom:
                self.err_up.append(self.v_est - self.v_odom)
            else:
                self.err_dn.append(self.v_odom - self.v_est)
        throttle = float(np.clip(v_set * THROTTLE_PER_MPS, 0.0, 1.0))

        self.count += 1
        self.lim_counts[lim] = self.lim_counts.get(lim, 0) + 1
        self.v_sum += v
        beside = valid & (x > -0.25) & (x < 0.10) & (np.abs(self.angles) > 0.5)
        if np.any(beside):
            self.min_body = min(self.min_body,
                                float(np.min(np.abs(y[beside]))) - CAR_HALF_WIDTH)
        self.min_front = min(self.min_front, front)
        if front < 1.0:
            self.v_close = max(self.v_close, v)
        if self.count % LOG_EVERY == 0:
            src = "m" if measured else "e"
            self.get_logger().info(
                f"v={v:4.2f}{src}->{self.v_cmd:4.2f}m/s ({lim}) set={v_set:4.2f} "
                f"real={self.v_odom if self.v_odom is not None else float('nan'):4.2f} "
                f"thr={throttle:.3f} "
                f"steer={steering:+.2f} e={e:+.2f} psi={math.degrees(psi):+5.1f} "
                f"front={front:4.1f} path={front_path:4.1f} w={w:.2f} "
                f"gap={math.degrees(theta_gap):+5.1f} "
                f"dL={left[0] if left else float('nan'):.2f} "
                f"dR={right[0] if right else float('nan'):.2f} "
                f"side={min(cl, cr):.2f} hz={1.0 / dt_f:.1f}")
        if self.count % STATS_EVERY == 0:
            share = " ".join(f"{k}={100.0 * c / STATS_EVERY:.0f}%"
                             for k, c in sorted(self.lim_counts.items(),
                                                key=lambda kv: -kv[1]))
            self.get_logger().info(
                f"STATS last {STATS_EVERY} scans: avg v={self.v_sum / STATS_EVERY:.2f} "
                f"m/s | limiter share: {share} | closest wall: beside "
                f"{self.min_body:.2f} m, straight ahead {self.min_front:.2f} m | "
                f"max v with wall <1 m ahead: {self.v_close:.2f} m/s | "
                f"max scan gap {self.dt_peak:.2f} s")
            up = float(np.mean(self.err_up)) if self.err_up else 0.0
            dn = float(np.mean(self.err_dn)) if self.err_dn else 0.0
            self.get_logger().info(
                f"LAG real accel max {self.a_max:.1f} m/s2, real brake max "
                f"{self.d_max:.1f} m/s2 | est above real {100.0 * len(self.err_up) / STATS_EVERY:.0f}% "
                f"(avg {up:.2f}), real above est "
                f"{100.0 * len(self.err_dn) / STATS_EVERY:.0f}% (avg {dn:.2f}) | "
                f"boost {100.0 * self.n_boost / STATS_EVERY:.0f}%, "
                f"brake {100.0 * self.n_brake / STATS_EVERY:.0f}%")
            self.a_max = self.d_max = 0.0
            self.err_up, self.err_dn = [], []
            self.n_boost = self.n_brake = 0
            self.lim_counts = {}
            self.v_sum = 0.0
            self.min_body = RANGE_CAP
            self.min_front = RANGE_CAP
            self.v_close = 0.0

        self.publish(throttle, steering)

    def publish(self, throttle, steering):
        t, s = Float32(), Float32()
        t.data = float(throttle)
        s.data = float(steering)
        self.pub_throttle.publish(t)
        self.pub_steering.publish(s)


def main():
    apply_profile(sys.argv[1:])
    rclpy.init()
    node = WallFollowRacer()
    node.get_logger().info(
        f"Profile: {PROFILE} (top speed {V_MAX:.1f} m/s = throttle "
        f"{V_MAX * THROTTLE_PER_MPS:.3f}, A_LAT={A_LAT}, A_BRAKE={A_BRAKE_PLAN}) "
        f"odom plan={'on' if USE_ODOM_PLAN else 'off'} K_BOOST={K_BOOST} "
        f"K_BRAKE={K_BRAKE} anti-wobble={'on' if ANTI_WOBBLE else 'off'} "
        f"entry limit={'on' if ENTRY_LIMIT else 'off'} lead={ENTRY_LEAD_T}s")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            node.publish(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()