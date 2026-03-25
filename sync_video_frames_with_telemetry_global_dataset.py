"""
Clip Enrichment Pipeline: Frames + Synchronized Telemetry

This script enriches each extracted driving clip by generating:
1. A dedicated folder for the clip.
2. A telemetry CSV aligned to clip timestamps.
3. A sequence of extracted video frames.

For every run directory inside the dataset, the script processes clips found in:
- video_data/evasion_driving_videos/
- video_data/normal_driving_videos/

For each clip <clip_name>.mp4, the following structure is generated:

    <clip_name>/
    ├── <clip_name>.mp4        # original clip moved into its own folder
    ├── <clip_name>.csv        # synchronized telemetry for the clip
    └── frames/
        ├── 00000.jpg
        ├── 00001.jpg
        └── ...

Telemetry is obtained from the corresponding carState CSV file and
synchronized to frame timestamps using nearest-neighbor matching.

Requirements:
- ffmpeg must be installed and available in PATH.

Example:
    python sync_video_frames_with_telemetry_global_dataset.py \
        --dataset_dir /path/to/Dataset \
        --run_csv_dir run1=/path/to/run1/csvs \
        --run_csv_dir run2=/path/to/run2/csvs \
        --run_csv_dir run3=/path/to/run3/csvs \
        --run_csv_dir run4=/path/to/run4/csvs
"""

import os
import re
import shutil
import argparse
import subprocess
import pandas as pd


# DEFAULT CONFIGURATION

# Telemetry fields to preserve in each per-clip CSV file
CARSTATE_FIELDS = [
    "vEgo", "gas", "brake", "steeringAngleDeg", "steeringTorque",
    "aEgo", "yawRate", "gearShifter", "steeringRateDeg", "vEgoRaw",
    "standstill", "leftBlinker", "rightBlinker", "gasPressed",
    "brakePressed", "steeringPressed"
]


# ARGUMENT PARSING

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Enrich extracted clips with synchronized telemetry and video frames."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset root directory containing run1, run2, run3, etc."
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frame rate used to synchronize telemetry with extracted frames."
    )

    parser.add_argument(
        "--run_csv_dir",
        action="append",
        default=[],
        help=(
            "Mapping between run name and the directory containing its carState CSV files. "
            "Format: run_name=/path/to/csvs . "
            "This argument can be repeated multiple times, e.g. "
            "--run_csv_dir run1=/path/to/run1/csvs --run_csv_dir run2=/path/to/run2/csvs"
        )
    )

    return parser.parse_args()


def parse_run_csv_mapping(run_csv_dir_args):
    """
    Convert repeated --run_csv_dir arguments into a dictionary.

    Args:
        run_csv_dir_args (list[str]): List of strings with format 'runX=/path/to/csvs'.

    Returns:
        dict: Mapping {run_name: csv_directory}.
    """
    mapping = {}

    for item in run_csv_dir_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid --run_csv_dir value: '{item}'. Expected format: run_name=/path/to/csvs"
            )

        run_name, csv_dir = item.split("=", 1)
        run_name = run_name.strip()
        csv_dir = csv_dir.strip()

        if not run_name:
            raise ValueError(f"Invalid run name in --run_csv_dir: '{item}'")
        if not csv_dir:
            raise ValueError(f"Invalid CSV directory in --run_csv_dir: '{item}'")

        mapping[run_name] = csv_dir

    return mapping


# UTILITY FUNCTIONS

def check_ffmpeg():
    """
    Check whether ffmpeg is installed and available in PATH.

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


def parse_clip_name(clip_name):
    """
    Extract video_id, t_start, and t_end from the clip filename.

    The last two underscore-separated numeric tokens are assumed to be:
    - t_start
    - t_end

    Examples:
        80f94eb526c7a9ac_0000009a--9f194eb0e9--13_1_0.198_6.798.mp4
            -> video_id = 80f94eb526c7a9ac_0000009a--9f194eb0e9--13
               t_start  = 0.198
               t_end    = 6.798

        80f94eb526c7a9ac_0000009a--9f194eb0e9--1_normal_1_0.0_47.4.mp4
            -> video_id = 80f94eb526c7a9ac_0000009a--9f194eb0e9--1
               t_start  = 0.0
               t_end    = 47.4

    Args:
        clip_name (str): Clip filename.

    Returns:
        tuple[str | None, float | None, float | None]:
            Parsed (video_id, t_start, t_end), or (None, None, None) if parsing fails.
    """
    stem = clip_name.replace(".mp4", "")
    parts = stem.split("_")

    try:
        t_end = float(parts[-1])
        t_start = float(parts[-2])
    except (ValueError, IndexError):
        return None, None, None

    # The video_id always ends with the token containing '--<number>'
    match = re.match(r"^(.+--\d+)", stem)
    if match:
        video_id = match.group(1)
    else:
        # Fallback heuristic if the expected pattern is not found
        video_id = "_".join(parts[:-3])

    return video_id, t_start, t_end


def find_carstate_csv(video_id, csvs_dir):
    """
    Find the carState CSV associated with a given video_id.

    The function checks multiple possible filename conventions.

    Args:
        video_id (str): Base video identifier.
        csvs_dir (str): Directory containing carState CSV files.

    Returns:
        str or None: Path to the matching CSV file, or None if not found.
    """
    candidates = [
        video_id + "--qlog_carState.csv",
        video_id + "--rlog_carState.csv",
        video_id + "--qlog._carState.csv",
        video_id + "--rlog._carState.csv",
    ]

    for name in candidates:
        path = os.path.join(csvs_dir, name)
        if os.path.exists(path):
            return path

    return None


def load_carstate(csv_path):
    """
    Load a carState CSV and compute relative timestamps in seconds.

    The source CSV is expected to contain a 't' column with timestamps in nanoseconds.
    A new column 'timestamp_sec' is created relative to the first sample.

    Args:
        csv_path (str): Path to the carState CSV file.

    Returns:
        pandas.DataFrame: Telemetry dataframe sorted by relative timestamp.

    Raises:
        ValueError: If the CSV does not contain a 't' column.
    """
    df = pd.read_csv(csv_path)

    if "t" not in df.columns:
        raise ValueError(f"CSV file does not contain required 't' column: {csv_path}")

    df["timestamp_sec"] = (df["t"] - df["t"].iloc[0]) / 1e9
    return df.sort_values("timestamp_sec").reset_index(drop=True)


def extract_telemetry_for_clip(carstate_df, t_start, t_end, fps):
    """
    Extract and synchronize telemetry rows for a clip interval.

    A regular sequence of frame timestamps is created using the provided FPS.
    Then, telemetry values are assigned to each frame timestamp using nearest-neighbor
    matching with pandas.merge_asof.

    Args:
        carstate_df (pandas.DataFrame): Full carState dataframe for the source video.
        t_start (float): Clip start time in seconds.
        t_end (float): Clip end time in seconds.
        fps (int): Frame rate used to generate frame timestamps.

    Returns:
        pandas.DataFrame: Synchronized telemetry dataframe with:
            - frame_id
            - selected telemetry fields
    """
    duration = t_end - t_start
    n_frames = max(1, round(duration * fps))

    # Frame timestamps expected for this clip
    frame_ts = [t_start + i / fps for i in range(n_frames)]
    df_frames = pd.DataFrame({"timestamp_sec": frame_ts})

    # Add a small margin around the clip interval to improve nearest-neighbor matching
    margin = 1.0
    cs_clip = carstate_df[
        (carstate_df["timestamp_sec"] >= t_start - margin) &
        (carstate_df["timestamp_sec"] <= t_end + margin)
    ].copy()

    cols = ["timestamp_sec"] + [f for f in CARSTATE_FIELDS if f in cs_clip.columns]

    df_sync = pd.merge_asof(
        df_frames.sort_values("timestamp_sec"),
        cs_clip[cols],
        on="timestamp_sec",
        direction="nearest",
        tolerance=0.5
    )

    # Add a local frame identifier consistent with the extracted image names
    df_sync.insert(0, "frame_id", [f"{i:05d}.jpg" for i in range(len(df_sync))])
    df_sync = df_sync.drop(columns=["timestamp_sec"])

    return df_sync


def extract_frames_from_clip(clip_path, frames_dir):
    """
    Extract all frames from a clip as JPEG images, numbered from 00000.jpg.

    Args:
        clip_path (str): Path to the clip file.
        frames_dir (str): Output directory for frames.

    Returns:
        bool: True if extraction succeeds, False otherwise.
    """
    os.makedirs(frames_dir, exist_ok=True)
    out_pattern = os.path.join(frames_dir, "%05d.jpg")

    cmd = [
        "ffmpeg",
        "-i", clip_path,
        "-start_number", "0",
        "-q:v", "2",
        "-y",
        out_pattern
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"        FFmpeg error: {e.stderr.decode()[:200]}")
        return False


def process_clip(clip_name, clips_dir, csvs_dir, fps):
    """
    Process one individual clip.

    Steps:
    1. Parse video_id, t_start, and t_end from the clip name.
    2. Locate and load the corresponding carState CSV.
    3. Create a dedicated folder for the clip.
    4. Move the .mp4 file into that folder.
    5. Generate a synchronized telemetry CSV.
    6. Extract all clip frames as JPEG images.

    Args:
        clip_name (str): Clip filename.
        clips_dir (str): Directory containing the clip.
        csvs_dir (str): Directory containing run-level carState CSVs.
        fps (int): Frame rate used for telemetry synchronization.

    Returns:
        str: One of:
            - "ok"        -> processed successfully
            - "skip"      -> already processed
            - "discarded" -> clip removed because no matching telemetry CSV was found
            - "error"     -> processing failed
    """
    clip_stem = clip_name.replace(".mp4", "")
    clip_folder = os.path.join(clips_dir, clip_stem)
    new_mp4 = os.path.join(clip_folder, clip_name)

    # Check whether the clip has already been fully processed
    if os.path.isdir(clip_folder) and os.path.exists(new_mp4):
        csv_path = os.path.join(clip_folder, f"{clip_stem}.csv")
        frames_dir = os.path.join(clip_folder, "frames")

        if (
            os.path.exists(csv_path)
            and os.path.exists(frames_dir)
            and len(os.listdir(frames_dir)) > 0
        ):
            print(f"      Already processed, skipping: {clip_name}")
            return "skip"

    # Parse clip metadata from filename
    video_id, t_start, t_end = parse_clip_name(clip_name)
    if video_id is None:
        print(f"      ERROR: could not parse clip name: {clip_name}")
        return "error"

    print(f"      {clip_name}")
    print(f"        video_id : {video_id}")
    print(f"        interval : {t_start}s -> {t_end}s ({round(t_end - t_start, 2)}s)")

    # Locate the corresponding telemetry CSV
    carstate_csv_path = find_carstate_csv(video_id, csvs_dir)
    if carstate_csv_path is None:
        print(f"        DISCARDED: no telemetry CSV found for '{video_id}', deleting clip.")
        clip_path = os.path.join(clips_dir, clip_name)

        if os.path.exists(clip_path):
            os.remove(clip_path)
            print(f"        Deleted clip: {clip_name}")

        return "discarded"

    try:
        carstate_df = load_carstate(carstate_csv_path)
    except Exception as e:
        print(f"        ERROR loading carState CSV: {e}")
        return "error"

    # Create clip folder and move the .mp4 file into it
    os.makedirs(clip_folder, exist_ok=True)

    clip_path = os.path.join(clips_dir, clip_name)
    if not os.path.exists(new_mp4):
        shutil.move(clip_path, new_mp4)

    clip_path = new_mp4

    # Generate synchronized telemetry CSV
    csv_path = os.path.join(clip_folder, f"{clip_stem}.csv")
    if not os.path.exists(csv_path):
        clip_tel = extract_telemetry_for_clip(carstate_df, t_start, t_end, fps)
        clip_tel.to_csv(csv_path, index=False)
        print(f"        Telemetry CSV saved: {len(clip_tel)} rows")
    else:
        print("        Telemetry CSV already exists")

    # Extract frames
    frames_dir = os.path.join(clip_folder, "frames")
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        if extract_frames_from_clip(clip_path, frames_dir):
            n_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
            print(f"        Frames extracted: {n_frames}")
        else:
            return "error"
    else:
        print("        Frames already extracted")

    return "ok"


# MAIN PIPELINE

def process_dataset(dataset_dir, run_csv_mapping, fps):
    """
    Process the full dataset across all run directories.

    For each runX directory:
    - process clips in evasion_driving_videos/
    - process clips in normal_driving_videos/

    Args:
        dataset_dir (str): Root dataset directory containing run folders.
        run_csv_mapping (dict): Mapping from run name to directory containing its CSV files.
        fps (int): Frame rate used for synchronization.
    """
    if not check_ffmpeg():
        return

    run_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if re.match(r"run\d+$", d) and os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not run_dirs:
        print(f"No run directories found in: {dataset_dir}")
        return

    print(f"Detected runs: {run_dirs}\n")

    for run_name in run_dirs:
        run_path = os.path.join(dataset_dir, run_name)
        evasions_dir = os.path.join(run_path, "video_data", "evasion_driving_videos")
        normal_dir = os.path.join(run_path, "video_data", "normal_driving_videos")

        # Retrieve the CSV directory corresponding to the current run
        csvs_dir = run_csv_mapping.get(run_name)
        if csvs_dir is None:
            print(f"WARNING: no CSV directory configured for {run_name}, skipping.")
            continue

        if not os.path.exists(csvs_dir):
            print(f"ERROR: configured CSV directory does not exist for {run_name}: {csvs_dir}")
            continue

        print(f"{'=' * 60}")
        print(f"Processing {run_name}")
        print(f"  CSV directory: {csvs_dir}")

        counters = {"ok": 0, "skip": 0, "error": 0, "discarded": 0}

        for clip_type, clips_dir in [("EVASION", evasions_dir), ("NORMAL", normal_dir)]:
            if not os.path.exists(clips_dir):
                print(f"  Directory not found: {clips_dir}")
                continue

            mp4_files = sorted([
                f for f in os.listdir(clips_dir)
                if f.endswith(".mp4") and os.path.isfile(os.path.join(clips_dir, f))
            ])

            print(f"\n  [{clip_type}] {len(mp4_files)} clips in {os.path.basename(clips_dir)}/")

            for clip_name in mp4_files:
                result = process_clip(clip_name, clips_dir, csvs_dir, fps)
                counters[result] += 1

        print(f"\n  Summary for {run_name}:")
        print(f"    Processed: {counters['ok']}")
        print(f"    Skipped:   {counters['skip']}")
        print(f"    Discarded: {counters['discarded']}")
        print(f"    Errors:    {counters['error']}")

    print(f"\n{'=' * 60}")
    print("Pipeline completed.")
    print("\nGenerated structure per clip:")
    print("  <clip_name>/")
    print("  ├── <clip_name>.mp4")
    print("  ├── <clip_name>.csv")
    print("  └── frames/")
    print("      ├── 00000.jpg")
    print("      └── ...")


# ENTRY POINT

if __name__ == "__main__":
    args = parse_args()
    run_csv_mapping = parse_run_csv_mapping(args.run_csv_dir)

    print("=" * 60)
    print("CLIP ENRICHMENT PIPELINE (frames + synchronized telemetry)")
    print("=" * 60)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"FPS:               {args.fps}")
    print("Run CSV mapping:")
    for run_name, csv_dir in run_csv_mapping.items():
        print(f"  {run_name} -> {csv_dir}")
    print()

    process_dataset(
        dataset_dir=args.dataset_dir,
        run_csv_mapping=run_csv_mapping,
        fps=args.fps
    )