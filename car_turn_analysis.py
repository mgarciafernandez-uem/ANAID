"""
Detects and characterizes evasive steering maneuvers (obstacle avoidances) from
CAN-Bus telemetry recorded by an ADAS-equipped vehicle.

The script reads one or more `carState` CSV files, each containing time-series
steering data, and applies two complementary detection algorithms:

  1. **Hysteresis pair detector** — finds classic double-peak evasion patterns
     (steer left --> steer right) using a hysteresis state machine.
  2. **Isolated peak detector**  — finds sharp, prominent single-peak maneuvers
     that do not fit the double-peak pattern (e.g. late, hard avoidances).

Detections from both methods are merged, deduplicated, and filtered to remove
common false positives (e.g. "pre-mountain" artefacts caused by the signal
rising into a much larger maneuver).

Output (one sub-folder per recording segment)
  _turn_analysis_out/
      <segment>__turns.csv    <- per-evasion statistics table
      <segment>__turns.png    <- steering angle plot with annotated detections
      _ALL_TURNS.csv          <- aggregated table across all segments

Usage
-----
  Set CARSTATE_DIR to the folder containing your *_carState.csv files, then:

      python car_turn_analysis.py

Requirements: numpy, pandas, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# CONFIGURATION


# Directory containing the CAN-Bus carState CSV files to analyse
CARSTATE_DIR = r"C:./csvs"

# All output files (per-segment CSVs, PNGs, and the global summary) are written here
OUT_DIR = os.path.join(CARSTATE_DIR, "_turn_analysis_out")



# Hysteresis thresholds

# Using two separate thresholds (ON > OFF) prevents noisy signals from
# triggering hundreds of spurious micro-events.  Think of it as a Schmitt
# trigger: the event starts when the angle exceeds ANGLE_ON and ends only
# when it drops below the lower ANGLE_OFF level.
#
#   Angle (°)
#   15 | ─ ─ ─ ─ ─ ─ ─ ─ ─  <- ANGLE_ON  (event starts)
#   10 | ─ ─ ─ ─ ─ ─ ─ ─ ─  <- ANGLE_OFF (event ends)

ANGLE_ON_DEG  = 10.0   # Steering angle (°) that opens a new turn event
ANGLE_OFF_DEG =  6.0   # Steering angle (°) that closes a turn event



# Total evasion duration constraints

MIN_TOTAL_EVASION_SEC    = 2.0   # An evasion must last at least this long (seconds)
MAX_TOTAL_EVASION_SEC    = 8.0   # Evasions longer than this are likely normal curves
ENFORCE_TOTAL_DURATION   = True  # Set to False to disable these duration filters



# Gap between the two sub-peaks of a double-peak evasion

# If the second steering pulse starts more than MAX_GAP_BETWEEN_PEAKS_SEC
# after the first one ends, they are treated as independent events.
MAX_GAP_BETWEEN_PEAKS_SEC = 2.00   # seconds



# First-peak (P1) filters — applied inside the hysteresis pair detector

P1_MIN_SUBTURN_SEC          =  0.15   # P1 must last at least this long (s)
P1_MAX_SUBTURN_SEC          =  4.50   # P1 must last at most this long (s)
P1_PEAK_MIN_ABS_ANGLE_DEG   = 40.0   # P1 peak must exceed this angle (°)
P1_PEAK_MAX_ABS_ANGLE_DEG   = 140.0  # P1 peak must stay below this angle (°)



# Isolated-peak detector parameters


# Amplitude range for a valid isolated evasion peak
PEAK_MIN_ABS_DEG = 65.0    # Peak must exceed this absolute angle (°)
PEAK_MAX_ABS_DEG = 180.0   # Physical upper limit of the steering sensor (°)

# Duration of the peak above its base threshold
PEAK_MIN_DUR_SEC = 1.5     # Peak must last at least this long (s)
PEAK_MAX_DUR_SEC = 5.0     # Peak must last at most this long (s)

# Baseline level — defines "normal driving" vs. a special maneuver.
# The peak must rise clearly above this level on both sides (isolation check).
#
#   Angle (°)
#   100 |        /\         <- PEAK (evasion)
#    80 |       /  \
#    65 | ─────/────\─────  <- BASELINE_MAX_DEG (isolation threshold)
#    40 |    /        \
#     0 |___/          \___
#            Time (s)
#
# If the signal never drops below BASELINE_MAX_DEG around the candidate peak,
# it means multiple consecutive turns are merging — not an isolated evasion.
BASELINE_MAX_DEG = 65.0

# Minimum prominence: how much the peak must stand out above its surroundings.
# Low prominence --> gradual curve; high prominence --> sharp evasion.
MIN_PROMINENCE_DEG = 20.0

# Smoothing window applied to the steering signal before peak detection.
# Removes sensor noise without distorting the shape of genuine maneuvers.
#
#   Before smoothing:  80 |  /\/\  <- spurious noise spikes
#   After  smoothing:  80 |  /──\  <- clean peak
SMOOTH_SEC = 0.2   # seconds

# Prominence window: how far to look left and right from a candidate peak
# when computing the local minimum used for prominence.
#
#   |<-- 2.5 s -->|<- 2.5 s -->|
#                 /\
#    ──────      /  \      ──
#          \────/    \────/
PROM_WIN_SEC = 2.5   # seconds

# Minimum average rise and fall rate of the peak (°/s).
# Slow peaks are gradual curves; fast peaks are sharp evasions.
MIN_RISE_RATE_DEG_PER_SEC = 25.0
MIN_FALL_RATE_DEG_PER_SEC = 25.0

# Width of the peak measured at PEAK_WIDTH_LEVEL_FRAC of its maximum height.
# A wide peak at 70% of its height indicates a slow, sustained turn (curve),
# not a quick evasive flick.
#
#   100 |      /\         <- peak height
#    70 |    _/  \_       <- measurement level (70%)
#    40 |   /      \
#     0 |__/________\__
PEAK_WIDTH_LEVEL_FRAC = 0.70   # Fraction of peak height at which width is measured
MAX_PEAK_WIDTH_SEC    = 2.0    # Maximum allowed peak width at that level (s)

# Sharpness = peak height / width at PEAK_WIDTH_LEVEL_FRAC.
# High sharpness --> concentrated, quick turn --> evasion.
# Low  sharpness --> spread-out, slow turn   --> normal curve.
#
#   SHARP (evasion): 100° / 1.5 s = 66.7 °/s   
#   WIDE  (curve):   100° / 5.0 s = 20.0 °/s   
MIN_SHARPNESS_DEG_PER_SEC = 40.0


# Optional extra filters for P1 (disabled by default)

USE_EXTRA_FILTERS       = False   # Enable to additionally require steer rate and torque
MIN_MAX_STEER_RATE_DEG  = 20.0    # Minimum peak steering rate (°/s) when enabled
MIN_MAX_TORQUE          = 150.0   # Minimum peak torque (Nm) when enabled


# "Pre-mountain" false-positive filter

# Some recordings contain a small precursor steering pulse immediately before
# a very large maneuver (the "big mountain").  Without this filter that small
# pulse would be incorrectly labelled as an evasion.
# The filter discards a detection if:
#   (a) its own peak is below PREMOUNTAIN_SMALL_MAX_DEG, AND
#   (b) a much larger peak (≥ BIG_MOUNTAIN_MIN_DEG) starts within
#       PREMOUNTAIN_MAX_GAP_SEC of the detection's end.

BIG_MOUNTAIN_MIN_DEG      = 150.0   # Minimum peak angle for the "big mountain" (°)
PREMOUNTAIN_LOOKAHEAD_SEC =   4.0   # How far ahead (s) to search for the big mountain
PREMOUNTAIN_MAX_GAP_SEC   =   1.2   # Maximum gap (s) between the small and big peaks
PREMOUNTAIN_SMALL_MAX_DEG = 140.0   # Precursor peaks above this threshold are kept


# DATA LOADING

def load_carstate(csv_path: str) -> pd.DataFrame:
    """
    Load a CAN-Bus carState CSV and prepare it for analysis.

    Steps performed:
      - Validates that a nanosecond timestamp column 't' is present.
      - Creates 'timestamp_sec': relative time in seconds from the first sample.
      - Ensures that all required signal columns exist; missing ones are filled
        with NaN so downstream code does not need extra null checks.
      - Sorts by time and resets the integer index.

    Parameters
    ----------
    csv_path : str
        Path to the carState CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame indexed from 0 with columns:
        timestamp_sec, steeringAngleDeg, steeringRateDeg, steeringTorque, vEgo.

    Raises
    ------
    ValueError
        If the mandatory 't' column is absent from the CSV.
    """
    df = pd.read_csv(csv_path)
    if "t" not in df.columns:
        raise ValueError(f"{csv_path} is missing the required 't' (timestamp) column.")

    # Normalise nanosecond epoch timestamps to seconds relative to recording start
    df["timestamp_sec"] = (df["t"] - df["t"].iloc[0]) / 1e9

    # Create placeholder columns for any signal that was not logged
    for col in ["steeringAngleDeg", "steeringRateDeg", "steeringTorque", "vEgo"]:
        if col not in df.columns:
            df[col] = np.nan

    return df.sort_values("timestamp_sec").reset_index(drop=True)


# HELPER UTILITIES

def event_duration_sec(df: pd.DataFrame, s: int, e: int) -> float:
    """
    Return the duration in seconds of the segment from row index s to row e.

    Used to reject events that are too short (likely sensor noise, < 0.15 s)
    or too long (likely a sustained curve rather than an evasion, > 8 s).
    """
    return float(df.loc[e, "timestamp_sec"] - df.loc[s, "timestamp_sec"])


def peak_abs_angle(angle_segment: np.ndarray) -> float:
    """
    Return the maximum absolute steering angle within a segment.

    A small peak (e.g. 30°) suggests a gentle lane change; a large peak
    (e.g. 90°) is characteristic of a hard evasive maneuver.
    Returns NaN if the segment is empty.
    """
    if angle_segment.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(angle_segment)))



# PRE-MOUNTAIN FALSE-POSITIVE FILTER


def is_premountain_false_positive(df: pd.DataFrame, s: int, e: int) -> bool:
    """
    Return True if the detection (s, e) is a small precursor pulse that
    immediately precedes a much larger steering maneuver and should be discarded.

    A "pre-mountain" artefact looks like this in the steering signal:

        Angle (°)
        160 |              /\\         <- BIG mountain (real maneuver)
        140 |             /  \\
        100 |   /\\       /    \\
         80 |  /  \\     /      \\
         60 | /    \\___/        \\___
         40 |/
            t_s   t_e  <- small detection (false positive)
                    |gap|

    The filter checks:
      1. The detected peak is below PREMOUNTAIN_SMALL_MAX_DEG.
      2. Within PREMOUNTAIN_LOOKAHEAD_SEC after the detection ends, a peak
         of at least BIG_MOUNTAIN_MIN_DEG appears.
      3. That big peak starts within PREMOUNTAIN_MAX_GAP_SEC of the detection
         end (i.e. the two pulses are directly connected).

    Parameters
    ----------
    df : pd.DataFrame
        Full telemetry DataFrame for the recording segment.
    s, e : int
        Start and end row indices of the candidate detection.

    Returns
    -------
    bool
        True --> discard the detection; False --> keep it.
    """
    t = df["timestamp_sec"].to_numpy(dtype=float)
    a = np.abs(df["steeringAngleDeg"].to_numpy(dtype=float))

    if s < 0 or e >= len(df) or s >= e:
        return False

    # 1) Peak of the candidate (the "small" detection)
    small_peak = float(np.nanmax(a[s:e + 1]))
    if not np.isfinite(small_peak):
        return False

    # If the detection itself is already very large, it is not a pre-mountain
    if small_peak > PREMOUNTAIN_SMALL_MAX_DEG:
        return False

    t_end = float(t[e])

    # 2) Look ahead within PREMOUNTAIN_LOOKAHEAD_SEC for the big mountain
    right_mask = (t > t_end) & (t <= t_end + PREMOUNTAIN_LOOKAHEAD_SEC)
    idxs = np.where(right_mask)[0]
    if idxs.size == 0:
        return False

    right_angles = a[idxs]
    if float(np.nanmax(right_angles)) < BIG_MOUNTAIN_MIN_DEG:
        return False

    # 3) Verify the big mountain starts close enough to the small detection
    # (find the first sample that crosses ANGLE_ON_DEG after the detection ends)
    start_big = None
    for k in idxs:
        if a[k] >= ANGLE_ON_DEG:
            start_big = k
            break

    if start_big is None:
        return False

    gap = float(t[start_big] - t_end)
    return gap <= PREMOUNTAIN_MAX_GAP_SEC


# DETECTION METHOD 1 — HYSTERESIS PAIR DETECTOR

def detect_turn_events(df: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Detect all intervals where the steering wheel is significantly deflected
    using a two-threshold hysteresis state machine.

    The hysteresis mechanism avoids splitting a single physical maneuver into
    many micro-events due to sensor noise:
      - A new event opens  when |angle| ≥ ANGLE_ON_DEG.
      - The current event closes when |angle| ≤ ANGLE_OFF_DEG.

    Returns
    -------
    list of (start_idx, end_idx) tuples
        Each tuple is the closed interval [start, end] of row indices in df
        that belong to one turn event.
    """
    angle = df["steeringAngleDeg"].to_numpy(dtype=float)
    abs_a = np.abs(angle)

    events   = []
    in_turn  = False
    start    = None

    for i in range(len(df)):
        if not in_turn:
            if abs_a[i] >= ANGLE_ON_DEG:
                in_turn = True
                start   = i
        else:
            if abs_a[i] <= ANGLE_OFF_DEG:
                events.append((start, i))
                in_turn = False
                start   = None

    # Close any event that was still open at the end of the recording
    if in_turn and start is not None:
        events.append((start, len(df) - 1))

    return events


def passes_p1_filters(df: pd.DataFrame, s: int, e: int) -> bool:
    """
    Return True if the hysteresis event (s, e) is a plausible first steering
    pulse (P1) of a double-peak evasion maneuver.

    Filter summary:
    
      Criterion              Threshold          Rejects if …                   
    ----------------------------------------------------------------------------
    │ Duration             │ 0.15 s – 4.5 s   │ Too short (noise) or too long  │
    │ Peak angle           │ 40° – 140°       │ Too gentle or physically       │
    │                      │                  │ impossible                     │
    │ Steer rate (optional)│ > 20 °/s         │ Giro too slow (requires        │
    │ Torque     (optional)│ > 150 Nm         │ USE_EXTRA_FILTERS = True)      │
 
    """
    dur = event_duration_sec(df, s, e)
    if dur < P1_MIN_SUBTURN_SEC or dur > P1_MAX_SUBTURN_SEC:
        return False

    seg = df.iloc[s:e + 1]
    a   = seg["steeringAngleDeg"].to_numpy(dtype=float)
    p   = peak_abs_angle(a)
    if not np.isfinite(p):
        return False
    if p < P1_PEAK_MIN_ABS_ANGLE_DEG or p > P1_PEAK_MAX_ABS_ANGLE_DEG:
        return False

    if USE_EXTRA_FILTERS:
        rate       = seg["steeringRateDeg"].to_numpy(dtype=float)
        torque     = seg["steeringTorque"].to_numpy(dtype=float)
        max_rate   = np.nanmax(np.abs(rate))
        max_torque = np.nanmax(np.abs(torque))
        if max_rate < MIN_MAX_STEER_RATE_DEG and max_torque < MIN_MAX_TORQUE:
            return False

    return True


def detect_evasions_first_peak_only(df: pd.DataFrame) -> list[dict]:
    """
    Detect evasive maneuvers that follow the classic double-peak pattern:

        Angle (°)
        40 |      /\\              /\\
        20 |     /  \\            /  \\
         0 |____/    \\__________/    \\____
                [P1]    [gap]   [P2]
                |<──── full evasion ────>|

    Algorithm:
      1. Extract all hysteresis events.
      2. For each event that passes the P1 quality filters, scan forward
         for a second event (P2) within MAX_GAP_BETWEEN_PEAKS_SEC.
      3. If found and the combined P1+P2 duration is within
         [MIN_TOTAL_EVASION_SEC, MAX_TOTAL_EVASION_SEC], record the evasion.

    Returns
    -------
    list of dict
        Each dict describes one detected evasion with keys:
        method, s_total, e_total, s1, e1, s2, e2.
    """
    events = detect_turn_events(df)
    if len(events) < 2:
        return []

    t        = df["timestamp_sec"].to_numpy(dtype=float)
    evasions = []
    i        = 0

    while i < len(events) - 1:
        s1, e1 = events[i]

        # Skip events that do not qualify as a valid first peak
        if not passes_p1_filters(df, s1, e1):
            i += 1
            continue

        found = False
        j     = i + 1

        while j < len(events):
            s2, e2 = events[j]
            gap    = float(t[s2] - t[e1])

            # Stop searching if the next event is too far away in time
            if gap > MAX_GAP_BETWEEN_PEAKS_SEC:
                break

            total = float(t[e2] - t[s1])
            if ENFORCE_TOTAL_DURATION and not (MIN_TOTAL_EVASION_SEC <= total <= MAX_TOTAL_EVASION_SEC):
                j += 1
                continue

            evasions.append({
                "method":  "hysteresis_pair",
                "s_total": s1, "e_total": e2,
                "s1": s1, "e1": e1,
                "s2": s2, "e2": e2,
            })
            found = True
            break

        # Advance past the consumed pair, or just move one step forward
        i = j + 1 if found else i + 1

    return evasions



# DETECTION METHOD 2 — ISOLATED PEAK DETECTOR


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    Apply a simple box (uniform-weight) moving average to a 1-D signal.

    Using 'same' convolution mode preserves the original array length so that
    the smoothed signal remains aligned with the time axis sample-for-sample.

    Before smoothing:  80 |  /\\/\\  <- noise spikes
    After  smoothing:  80 |  /────\\  <- clean peak

    Parameters
    ----------
    x   : np.ndarray  Raw input signal.
    win : int         Window size in samples (win ≤ 1 returns x unchanged).
    """
    if win <= 1:
        return x
    kernel = np.ones(int(win)) / int(win)
    return np.convolve(x, kernel, mode="same")


def detect_isolated_peaks_improved(df: pd.DataFrame) -> list[dict]:
    """
    Detect sharp, prominent, isolated steering peaks that represent single-
    pulse evasive maneuvers not captured by the hysteresis pair detector.

    The algorithm evaluates each local maximum of the smoothed |steeringAngleDeg|
    signal against nine sequential quality criteria:

    │ #  │ Criterion                                                  │
    -------------------------------------------------------------------
    │ 1  │ Local maximum (a[i] ≥ a[i-1] and a[i] > a[i+1])            │
    │ 2  │ Amplitude within [PEAK_MIN_ABS_DEG, PEAK_MAX_ABS_DEG]      │
    │ 3  │ Base threshold = max(BASELINE_MAX_DEG, peak × 0.30)        │
    │ 4  │ Duration above base in [PEAK_MIN_DUR_SEC, PEAK_MAX_DUR_SEC]│
    │ 5  │ Prominence ≥ MIN_PROMINENCE_DEG                            │
    │ 6  │ Signal returns below BASELINE_MAX_DEG × 1.2 on both sides  │
    │ 7  │ Average rise/fall rate ≥ 70% of MIN_RISE_RATE_DEG_PER_SEC  │
    │ 8  │ Width at PEAK_WIDTH_LEVEL_FRAC ≤ MAX_PEAK_WIDTH_SEC        │
    │ 9  │ Sharpness (height/width) ≥ MIN_SHARPNESS_DEG_PER_SEC       │
    

    Duplicate detections that overlap in time are deduplicated, keeping the
    highest-amplitude peak for each overlapping cluster.

    Returns
    -------
    list of dict
        Each dict describes one isolated-peak detection with keys:
        method, s_total, e_total, s1, e1, s2, e2,
        peak_idx, peak_abs_deg, peak_width_sec,
        rise_rate_deg_s, fall_rate_deg_s, sharpness_deg_s,
        prominence, duration_sec.
    """
    t = df["timestamp_sec"].to_numpy(dtype=float)
    a = np.abs(df["steeringAngleDeg"].to_numpy(dtype=float))

    # Estimate the median sample interval (dt) to convert seconds --> samples
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.05
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.05

    smooth_n = max(1, int(round(SMOOTH_SEC  / dt)))
    prom_n   = max(5, int(round(PROM_WIN_SEC / dt)))

    a_s = moving_average(a, smooth_n)   # Smoothed absolute steering angle

    peaks = []

    for i in range(1, len(a_s) - 1):

        #  Criterion 1: local maximum 
        if not (a_s[i] >= a_s[i - 1] and a_s[i] > a_s[i + 1]):
            continue

        #  Criterion 2: amplitude range
        peak_val = float(a_s[i])
        if peak_val < PEAK_MIN_ABS_DEG or peak_val > PEAK_MAX_ABS_DEG:
            continue

        #  Criterion 3 & 4: duration above dynamic base threshold 
        # The threshold is 30% of the peak height or BASELINE_MAX_DEG,
        # whichever is larger.  This prevents very tall peaks from having
        # their base cut too low.
        threshold = max(BASELINE_MAX_DEG, peak_val * 0.3)

        left = i
        while left > 0 and a_s[left] > threshold:
            left -= 1

        right = i
        while right < len(a_s) - 1 and a_s[right] > threshold:
            right += 1

        dur = float(t[right] - t[left])
        if dur < PEAK_MIN_DUR_SEC or dur > PEAK_MAX_DUR_SEC:
            continue

        #  Criterion 5: prominence
        # Prominence = peak_val − min(signal in the surrounding window),
        # excluding the peak region itself.
        l2 = max(0, i - prom_n)
        r2 = min(len(a_s) - 1, i + prom_n)

        left_context  = a_s[l2:left]    if left  > l2 else []
        right_context = a_s[right:r2+1] if right < r2 else []

        if len(left_context) > 0 and len(right_context) > 0:
            local_min = float(min(np.nanmin(left_context), np.nanmin(right_context)))
        elif len(left_context) > 0:
            local_min = float(np.nanmin(left_context))
        elif len(right_context) > 0:
            local_min = float(np.nanmin(right_context))
        else:
            local_min = 0.0

        prom = float(peak_val - local_min)
        if prom < MIN_PROMINENCE_DEG:
            continue

        # Criterion 6: isolation check
        # The signal must settle below BASELINE_MAX_DEG × 1.2 on each side
        # of the peak (a small margin accounts for sensor noise at the edges).
        margin    = 3   # samples of tolerance at the boundary
        pre_vals  = a_s[max(0, left - margin):left + 1]
        post_vals = a_s[right:min(len(a_s), right + margin + 1)]

        pre_ok  = len(pre_vals)  == 0 or float(np.nanmean(pre_vals))  < BASELINE_MAX_DEG * 1.2
        post_ok = len(post_vals) == 0 or float(np.nanmean(post_vals)) < BASELINE_MAX_DEG * 1.2

        if not (pre_ok and post_ok):
            continue

        # Criterion 7: rise and fall rates 
        t_left  = float(t[left])
        t_peak  = float(t[i])
        t_right = float(t[right])

        rise_dt = max(1e-6, t_peak  - t_left)
        fall_dt = max(1e-6, t_right - t_peak)

        base_left  = float(a_s[left])
        base_right = float(a_s[right])

        rise_rate = (peak_val - base_left)  / rise_dt
        fall_rate = (peak_val - base_right) / fall_dt

        # Accept if the average rate meets 70% of the configured threshold
        # (slightly more permissive to capture asymmetric peaks)
        avg_rate = (rise_rate + fall_rate) / 2
        if avg_rate < MIN_RISE_RATE_DEG_PER_SEC * 0.7:
            continue

        #  Criterion 8: peak width at PEAK_WIDTH_LEVEL_FRAC 
        level = PEAK_WIDTH_LEVEL_FRAC * peak_val

        wl = i
        while wl > 0 and a_s[wl] >= level:
            wl -= 1

        wr = i
        while wr < len(a_s) - 1 and a_s[wr] >= level:
            wr += 1

        width_sec = float(t[wr] - t[wl])
        if width_sec > MAX_PEAK_WIDTH_SEC:
            continue

        #  Criterion 9: sharpness 
        # sharpness = height / width  [°/s]
        # High sharpness --> concentrated, quick turn --> evasion
        # Low  sharpness --> spread-out, slow turn   --> normal curve
        sharpness = peak_val / max(1e-6, width_sec)
        if sharpness < MIN_SHARPNESS_DEG_PER_SEC:
            continue

        peaks.append({
            "method":           "isolated_peak",
            "s_total":          left,
            "e_total":          right,
            "s1": left, "e1":   right,
            "s2": np.nan, "e2": np.nan,
            "peak_idx":         i,
            "peak_abs_deg":     peak_val,
            "peak_width_sec":   width_sec,
            "rise_rate_deg_s":  rise_rate,
            "fall_rate_deg_s":  fall_rate,
            "sharpness_deg_s":  sharpness,
            "prominence":       prom,
            "duration_sec":     dur,
        })


    # Deduplication: keep only the highest-amplitude peak per overlapping
    # cluster so that adjacent local maxima from the same maneuver are not
    # counted multiple times.

    peaks_sorted = sorted(peaks, key=lambda d: d["peak_abs_deg"], reverse=True)
    merged       = []

    for p in peaks_sorted:
        overlap = False
        for existing in merged:
            # Two detections overlap if their time intervals are not disjoint
            if not (p["e_total"] < existing["s_total"] or p["s_total"] > existing["e_total"]):
                overlap = True
                break
        if not overlap:
            merged.append(p)

    # Re-sort chronologically before returning
    return sorted(merged, key=lambda d: d["s_total"])



# DETECTION MERGING — combine results from both detectors


def iou_1d(a0, a1, b0, b1) -> float:
    """
    Compute the 1-D Intersection-over-Union (IoU) between two time intervals.

    IoU = |intersection| / |union|

    Example:
        Interval A: [════════]           (a0 --> a1)
        Interval B:      [════════]      (b0 --> b1)
                    ─────^^^^─────
        Intersection: ^^^^
        IoU = len(^^^^) / len(─────────)

    Returns 0.0 when the intervals are disjoint; 1.0 for identical intervals.
    """
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def merge_detections(df: pd.DataFrame, dets: list[dict]) -> list[dict]:
    """
    Merge detections from both the hysteresis-pair and isolated-peak detectors,
    collapsing pairs that substantially overlap in time into a single entry.

    Two detections are merged when their 1-D IoU ≥ 0.35 (i.e. they share at
    least 35% of their combined time span).  The merged entry spans the union
    of both intervals and carries a combined method label, e.g.
    "hysteresis_pair+peak".

    Example:
        Method A (hysteresis):  [════════]        5 s --> 8 s
        Method B (peak):             [════════]   6 s --> 9 s
        IoU = (8−6) / (9−5) = 0.50  ≥ 0.35  --> merge
        Result:                 [════════════]    5 s --> 9 s

    Parameters
    ----------
    df   : pd.DataFrame  Full telemetry DataFrame (used to convert indices --> seconds).
    dets : list of dict  Concatenated detections from both detectors.

    Returns
    -------
    list of dict
        Deduplicated and merged detections sorted by start time.
    """
    if not dets:
        return dets

    t    = df["timestamp_sec"].to_numpy(dtype=float)
    dets = sorted(dets, key=lambda d: (d["s_total"], d["e_total"]))

    out = []
    for d in dets:
        s = int(d["s_total"])
        e = int(d["e_total"])

        if not out:
            out.append(d)
            continue

        ps  = int(out[-1]["s_total"])
        pe  = int(out[-1]["e_total"])
        iou = iou_1d(float(t[s]), float(t[e]), float(t[ps]), float(t[pe]))

        if iou >= 0.35:
            # Expand the existing entry to cover both detections
            out[-1]["s_total"] = min(ps, s)
            out[-1]["e_total"] = max(pe, e)
            out[-1]["method"]  = out[-1]["method"] + "+peak"
        else:
            out.append(d)

    return out


# SUMMARY AND VISUALISATION

def summarize(df: pd.DataFrame, detections: list[dict], video_name: str) -> pd.DataFrame:
    """
    Build a per-evasion statistics table from the list of detections.

    For each detected evasion the following metrics are computed over the
    segment df[s_total : e_total]:


    │ Column                    │ Description                              │
    ------------------------------------------------------------------------
    │ video                     │ Recording segment identifier             │
    │ evasion_id                │ Sequential 1-based index per segment     │
    │ method                    │ Detection method label                   │
    │ t_start / t_end           │ Absolute time within the recording (s)   │
    │ duration_sec              │ t_end − t_start                          │
    │ max/mean_abs_angle_deg    │ Peak and average steering angle (°)      │
    │ max/mean_abs_steer_rate   │ Peak and average steer rate (°/s)        │
    │ max/mean_abs_torque       │ Peak and average steering torque (Nm)    │
    │ mean/min_vEgo             │ Vehicle speed statistics (m/s)           │


    Returns
    -------
    pd.DataFrame
        One row per detected evasion.
    """
    rows = []
    for k, ev in enumerate(detections, start=1):
        sT, eT = int(ev["s_total"]), int(ev["e_total"])
        t0  = float(df.loc[sT, "timestamp_sec"])
        t1  = float(df.loc[eT, "timestamp_sec"])
        seg = df.iloc[sT:eT + 1]

        a      = seg["steeringAngleDeg"].to_numpy(dtype=float)
        rate   = seg["steeringRateDeg"].to_numpy(dtype=float)
        torque = seg["steeringTorque"].to_numpy(dtype=float)
        vego   = seg["vEgo"].to_numpy(dtype=float)

        rows.append({
            "video":                    video_name,
            "evasion_id":               k,
            "method":                   ev.get("method", "unknown"),
            "t_start":                  t0,
            "t_end":                    t1,
            "duration_sec":             t1 - t0,
            "max_abs_angle_deg":        float(np.nanmax(np.abs(a))),
            "mean_abs_angle_deg":       float(np.nanmean(np.abs(a))),
            "max_abs_steer_rate_deg":   float(np.nanmax(np.abs(rate))),
            "mean_abs_steer_rate_deg":  float(np.nanmean(np.abs(rate))),
            "max_abs_torque":           float(np.nanmax(np.abs(torque))),
            "mean_abs_torque":          float(np.nanmean(np.abs(torque))),
            "mean_vEgo":  float(np.nanmean(vego)) if np.isfinite(np.nanmean(vego)) else np.nan,
            "min_vEgo":   float(np.nanmin(vego))  if np.isfinite(np.nanmin(vego))  else np.nan,
        })

    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, detections: list[dict], out_png: str, title: str):
    """
    Save a time-series plot of the absolute steering angle with each detected
    evasion highlighted as a shaded vertical span.

    The ANGLE_ON and ANGLE_OFF threshold lines are drawn as horizontal dashed
    references so the hysteresis behaviour is immediately visible.

    Parameters
    ----------
    df         : pd.DataFrame   Telemetry for the recording segment.
    detections : list of dict   Merged detections to annotate.
    out_png    : str            Output file path for the PNG image.
    title      : str            Plot title (typically the segment name + count).
    """
    t     = df["timestamp_sec"].to_numpy(dtype=float)
    angle = np.abs(df["steeringAngleDeg"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, angle, label="|steeringAngleDeg|", linewidth=1.0, alpha=0.9)
    ax.axhline(ANGLE_ON_DEG,  linestyle="--", linewidth=1.5, label=f"ANGLE_ON={ANGLE_ON_DEG}")
    ax.axhline(ANGLE_OFF_DEG, linestyle="--", linewidth=1.0, label=f"ANGLE_OFF={ANGLE_OFF_DEG}")

    # Shade each detected evasion window
    for ev in detections:
        sT, eT = int(ev["s_total"]), int(ev["e_total"])
        ax.axvspan(df.loc[sT, "timestamp_sec"], df.loc[eT, "timestamp_sec"], alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering angle (°)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


# MAIN

def main():
    """
    Entry point: iterate over all carState CSVs in CARSTATE_DIR, run both
    detectors on each, apply the pre-mountain false-positive filter, and
    write per-segment output files plus a global summary CSV.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect all carState CSV files in the configured directory
    csvs = [f for f in os.listdir(CARSTATE_DIR) if f.endswith(".csv") and "carState" in f]
    if not csvs:
        print(f"No carState CSVs found in: {CARSTATE_DIR}")
        return

    all_rows = []

    for fname in sorted(csvs):
        path       = os.path.join(CARSTATE_DIR, fname)
        video_name = fname.replace("--qlog_carState.csv", "").replace("--rlog_carState.csv", "")

        try:
            df = load_carstate(path)
        except Exception as e:
            print(f"{fname}: error loading ({e})")
            continue

        # Run both detectors and combine their results
        det_a = detect_evasions_first_peak_only(df)     # double-peak (hysteresis)
        det_b = detect_isolated_peaks_improved(df)      # single isolated peak
        detections = merge_detections(df, det_a + det_b)

        # Remove false positives caused by small pre-mountain precursor pulses
        detections = [
            d for d in detections
            if not is_premountain_false_positive(df, int(d["s_total"]), int(d["e_total"]))
        ]

        # Summarise and save per-segment results
        summary = summarize(df, detections, video_name)
        summary.to_csv(os.path.join(OUT_DIR, f"{video_name}__turns.csv"), index=False)
        plot(df, detections, os.path.join(OUT_DIR, f"{video_name}__turns.png"),
             title=f"{video_name} | evasions detected: {len(summary)}")

        all_rows.append(summary)

        if len(summary) == 0:
            print(f" {video_name}: 0 evasions detected")
        else:
            print(f" {video_name}: {len(summary)} evasions "
                  f"| first: {summary.loc[0,'t_start']:.1f} s – {summary.loc[0,'t_end']:.1f} s")

    # Aggregate all segments into a single global CSV
    if all_rows:
        global_df   = pd.concat(all_rows, ignore_index=True)
        global_path = os.path.join(OUT_DIR, "_ALL_TURNS.csv")
        global_df.to_csv(global_path, index=False)
        print(f"\nGlobal summary saved: {global_path}")

    print(f"\nOutput directory: {OUT_DIR}")


if __name__ == "__main__":
    main()