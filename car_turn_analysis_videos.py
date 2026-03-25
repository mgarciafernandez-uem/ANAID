"""
Video Fragment Extraction Pipeline for Driving Behavior Analysis

This script extracts video clips corresponding to:
1. Evasion maneuvers ("evasions") detected in a CSV file.
2. Normal driving segments, defined as the complementary intervals
   not covered by any evasion segment.

Input:
- A CSV file containing detected evasions with start/end timestamps.
- A directory containing the original driving videos.

Output:
- OUTPUT_DIR/evasions/ -> clips containing evasive maneuvers
- OUTPUT_DIR/normal/   -> clips containing normal driving

Requirements:
- ffmpeg and ffprobe must be installed and available in PATH.

Example:
    python car_turn_analysis_videos.py --base_dir /path/to/project

Optional explicit paths:
    python car_turn_analysis_videos.py ^
        --base_dir C:/path/to/project ^
        --csv_file C:/path/to/_ALL_TURNS.csv ^
        --videos_dir C:/path/to/videos ^
        --output_dir C:/path/to/output
"""

import os
import argparse
import subprocess
import pandas as pd


# ARGUMENT PARSING

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract evasion and normal-driving video fragments from annotated turn intervals."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Root directory of the project."
    )

    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Path to the CSV file containing evasion intervals. "
             "If not provided, a default path relative to base_dir is used."
    )

    parser.add_argument(
        "--videos_dir",
        type=str,
        default=None,
        help="Directory containing the original video files. "
             "If not provided, a default path relative to base_dir is used."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where extracted clips will be saved. "
             "If not provided, a default path relative to base_dir is used."
    )

    parser.add_argument(
        "--min_normal_duration",
        type=float,
        default=2.0,
        help="Minimum duration (in seconds) for a normal-driving segment to be saved."
    )

    return parser.parse_args()


# UTILITY FUNCTIONS

def check_ffmpeg():
    """
    Check whether ffmpeg is installed and accessible from the system PATH.

    Returns:
        bool: True if ffmpeg is available, False otherwise.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg is not installed or not available in PATH.")
        return False


def get_video_duration(video_path):
    """
    Get the duration of a video using ffprobe.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        float or None: Video duration in seconds, or None if the duration
        could not be retrieved.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return float(result.stdout.decode().strip())
    except Exception:
        return None


def extract_video_chunk(input_video, output_video, t_start, t_end):
    """
    Extract a video segment using ffmpeg without re-encoding.

    Args:
        input_video (str): Path to the original video.
        output_video (str): Output path for the extracted clip.
        t_start (float): Segment start time in seconds.
        t_end (float): Segment end time in seconds.

    Returns:
        bool: True if extraction succeeds, False otherwise.
    """
    duration = round(t_end - t_start, 3)

    if duration <= 0:
        return False

    cmd = [
        "ffmpeg",
        "-ss", str(t_start),
        "-i", input_video,
        "-t", str(duration),
        "-c", "copy",
        "-y",
        output_video
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FFmpeg error: {e.stderr.decode()[:200]}")
        return False


def normal_intervals(evasion_intervals, video_duration, min_normal_duration_sec):
    """
    Compute complementary intervals corresponding to normal driving.

    Given a list of evasion intervals and the full video duration,
    this function returns the non-overlapping intervals not covered
    by any evasion segment.

    Args:
        evasion_intervals (list[tuple[float, float]]): List of (start, end) evasion intervals.
        video_duration (float): Total video duration in seconds.
        min_normal_duration_sec (float): Minimum allowed duration for normal-driving clips.

    Returns:
        list[tuple[float, float]]: List of (start, end) normal-driving intervals.
    """
    intervals = sorted(evasion_intervals, key=lambda x: x[0])

    normal = []
    cursor = 0.0

    for t_start, t_end in intervals:
        if cursor < t_start:
            normal.append((round(cursor, 3), round(t_start, 3)))
        cursor = max(cursor, t_end)

    if cursor < video_duration:
        normal.append((round(cursor, 3), round(video_duration, 3)))

    normal = [(s, e) for s, e in normal if (e - s) >= min_normal_duration_sec]

    return normal


def clean_video_id(raw_video_id):
    """
    Normalize the video identifier stored in the CSV.

    Some CSV entries may include suffixes such as:
    - --rlog._carState.csv
    - --rlog_carState.csv
    - --qlog._carState.csv
    - --qlog_carState.csv
    - .csv

    This function removes those suffixes to recover the base video ID.

    Args:
        raw_video_id (str): Raw video identifier from the CSV.

    Returns:
        str: Cleaned video identifier.
    """
    vid = str(raw_video_id)

    for suffix in [
        "--rlog._carState.csv",
        "--rlog_carState.csv",
        "--qlog._carState.csv",
        "--qlog_carState.csv"
    ]:
        if vid.endswith(suffix):
            vid = vid[:-len(suffix)]
            break

    if vid.endswith(".csv"):
        vid = os.path.splitext(vid)[0]

    return vid


# MAIN PIPELINE

def process_turns_csv(csv_file, videos_dir, output_dir, min_normal_duration_sec):
    """
    Main pipeline for extracting evasion and normal-driving clips.

    Workflow:
    1. Load the annotated CSV file.
    2. Group evasion records by video.
    3. Extract all evasion segments.
    4. Compute and extract normal-driving segments.
    5. Process videos with no evasions at all.
    6. Print a summary report.

    Args:
        csv_file (str): Path to the CSV file containing evasion intervals.
        videos_dir (str): Directory containing the original video files.
        output_dir (str): Directory where output clips will be saved.
        min_normal_duration_sec (float): Minimum duration for normal clips.
    """
    if not check_ffmpeg():
        return

    # Validate input resources
    for path, label in [(csv_file, "CSV file"), (videos_dir, "video directory")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            return

    # Create output folders
    evasions_dir = os.path.join(output_dir, "evasions")
    normal_dir = os.path.join(output_dir, "normal")
    os.makedirs(evasions_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    # Load CSV annotations
    print(f"Reading CSV file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"ERROR while reading CSV: {e}")
        return

    print(f"Total rows in CSV: {len(df)}")
    print(f"Columns found: {list(df.columns)}\n")

    # Normalize video IDs to match video filenames
    df["video_id_clean"] = df["video"].apply(clean_video_id)

    # Group annotations by video
    grouped = df.groupby("video_id_clean")
    all_video_ids = sorted(df["video_id_clean"].unique())

    counters = {
        "evasion_ok": 0,
        "evasion_skip": 0,
        "evasion_err": 0,
        "normal_ok": 0,
        "normal_skip": 0,
        "normal_err": 0,
    }

    
    # Process videos that contain one or more evasions
    
    for video_id in all_video_ids:
        video_filename = f"{video_id}--qcamera.ts"
        input_video_path = os.path.join(videos_dir, video_filename)

        if not os.path.exists(input_video_path):
            print(f"WARNING: video not found -> {video_filename}")
            continue

        video_duration = get_video_duration(input_video_path)
        rows = grouped.get_group(video_id)
        evasion_intervals = []

        print(f"\n{'=' * 60}")
        print(f"Video: {video_id}")
        print(f"Duration: {video_duration:.2f}s" if video_duration else "Duration: unknown")
        print(f"Detected evasions: {len(rows)}")

        # 1. Extract evasion segments
        for _, row in rows.iterrows():
            t_start = round(float(row["t_start"]), 3)
            t_end = round(float(row["t_end"]), 3)
            evasion_id = row["evasion_id"]

            evasion_intervals.append((t_start, t_end))

            out_name = f"{video_id}_{evasion_id}_{t_start}_{t_end}.mp4"
            out_path = os.path.join(evasions_dir, out_name)

            if os.path.exists(out_path):
                print(f"  [EVASION] Already exists, skipping: {out_name}")
                counters["evasion_skip"] += 1
                continue

            print(f"  [EVASION] {t_start}s -> {t_end}s ({round(t_end - t_start, 2)}s)")
            if extract_video_chunk(input_video_path, out_path, t_start, t_end):
                print(f"    Saved: {out_name}")
                counters["evasion_ok"] += 1
            else:
                print("    Extraction failed")
                counters["evasion_err"] += 1

        # 2. Extract normal-driving segments
        if video_duration is None:
            # Fallback: estimate the duration from the maximum CSV end time
            video_duration = round(float(rows["t_end"].max()) + 1.0, 3)

        normal_segs = normal_intervals(
            evasion_intervals=evasion_intervals,
            video_duration=video_duration,
            min_normal_duration_sec=min_normal_duration_sec
        )

        print(f"Normal driving segments: {len(normal_segs)}")

        for seg_idx, (ns, ne) in enumerate(normal_segs, start=1):
            out_name = f"{video_id}_normal_{seg_idx}_{ns}_{ne}.mp4"
            out_path = os.path.join(normal_dir, out_name)

            if os.path.exists(out_path):
                print(f"  [NORMAL] Already exists, skipping: {out_name}")
                counters["normal_skip"] += 1
                continue

            print(f"  [NORMAL] {ns}s -> {ne}s ({round(ne - ns, 2)}s)")
            if extract_video_chunk(input_video_path, out_path, ns, ne):
                print(f"    Saved: {out_name}")
                counters["normal_ok"] += 1
            else:
                print("    Extraction failed")
                counters["normal_err"] += 1

    
    # Process videos with no evasions
    
    print(f"\n{'=' * 60}")
    print("Searching for videos without any detected evasions...")

    all_ts_files = [f for f in os.listdir(videos_dir) if f.endswith("--qcamera.ts")]

    for ts_file in sorted(all_ts_files):
        video_id = ts_file.replace("--qcamera.ts", "")

        if video_id in all_video_ids:
            continue

        input_video_path = os.path.join(videos_dir, ts_file)
        video_duration = get_video_duration(input_video_path)

        if video_duration is None:
            print(f"  WARNING: could not determine duration for {ts_file}")
            continue

        if video_duration < min_normal_duration_sec:
            continue

        out_name = f"{video_id}_normal_1_0.0_{round(video_duration, 3)}.mp4"
        out_path = os.path.join(normal_dir, out_name)

        if os.path.exists(out_path):
            print(f"  [NORMAL] Already exists, skipping: {out_name}")
            counters["normal_skip"] += 1
            continue

        print(f"  [NORMAL NO EVASION] {video_id} (0.0s -> {video_duration:.2f}s)")
        if extract_video_chunk(input_video_path, out_path, 0.0, video_duration):
            print(f"    Saved: {out_name}")
            counters["normal_ok"] += 1
        else:
            print("    Extraction failed")
            counters["normal_err"] += 1

    
    # Final summary

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  EVASION clips -> OK: {counters['evasion_ok']} | "
        f"Error: {counters['evasion_err']} | Skipped: {counters['evasion_skip']}"
    )
    print(
        f"  NORMAL clips  -> OK: {counters['normal_ok']} | "
        f"Error: {counters['normal_err']} | Skipped: {counters['normal_skip']}"
    )
    print(f"\n  Evasion output directory: {evasions_dir}")
    print(f"  Normal output directory:  {normal_dir}")


# ENTRY POINT

if __name__ == "__main__":
    args = parse_args()

    # Build default paths relative to the user-provided base directory
    csv_file = args.csv_file or os.path.join(
        args.base_dir, "jsons_finales", "csvs", "_turn_analysis_out", "_ALL_TURNS.csv"
    )
    videos_dir = args.videos_dir or os.path.join(args.base_dir, "videos")
    output_dir = args.output_dir or os.path.join(
        args.base_dir, "jsons_finales", "csvs", "_turn_analysis_out"
    )

    print("=" * 60)
    print("VIDEO FRAGMENT EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"Base directory:         {args.base_dir}")
    print(f"CSV file:               {csv_file}")
    print(f"Videos directory:       {videos_dir}")
    print(f"Output directory:       {output_dir}")
    print(f"Min normal clip length: {args.min_normal_duration} s")
    print()

    process_turns_csv(
        csv_file=csv_file,
        videos_dir=videos_dir,
        output_dir=output_dir,
        min_normal_duration_sec=args.min_normal_duration
    )

    print("\nProcess completed.")