"""
Synchronizes video footage with busCAN telemetry data (steering torque)
to produce a labeled image dataset suitable for end-to-end driving models such as PilotNet.

Pipeline overview
  1. Read a busCAN CSV file that contains timestamped steering torque values.
  2. Extract every frame from the corresponding dashcam video (.ts) and save
     each one as a JPEG image.
  3. Align video frames with telemetry using a nearest-neighbor temporal merge
     (pandas merge_asof) within a configurable tolerance window.
  4. Write a `data.txt` annotation file pairing each valid frame filename with
     its corresponding steering torque value.

Expected input directory layout:
  videos_dir/
      <segment>--qcamera.ts <- raw dashcam video
  csvs_dir/
      <segment>--rlog_carState.csv  <- busCAN telemetry (one of several naming variants)

Output layout (one folder per video segment):
  output_root/
      <segment>/
          00000.jpg <- extracted frame 0
          00001.jpg
          ...
          data.txt <- "filename steeringTorque" per valid frame

Usage:
  Adjust the three path constants below (videos_dir, csvs_dir, output_root)
  to match your local filesystem, then run:

      python sync_videos_with_busCANData_pilotnet.py

Requirements: opencv-python, pandas, numpy
"""

import os
import cv2
import pandas as pd
import numpy as np



# CONFIGURATION

# Directory that contains the raw dashcam video files (.ts format)
videos_dir = r"./videos"

# Directory that contains the busCAN telemetry CSVs exported from the vehicle logs
csvs_dir = r"C:./csvs"

# Root output directory; one sub-folder will be created per processed video segment
output_root = r"./pilotnet_datasets"

# Fallback frame rate used when OpenCV cannot read the FPS metadata from the video
fps_default = 20

# Maximum time difference (in seconds) allowed between a video frame timestamp
# and its nearest busCAN sample during synchronization.
# Frames with no telemetry match within this window are discarded.
tolerance_sec = 0.1


os.makedirs(output_root, exist_ok=True)



# CORE PROCESSING FUNCTION

def procesar_video(video_path, carstate_csv):
    """
    Process a single video segment: extract frames, synchronize with busCAN
    telemetry, and write the annotation file.

    Parameters
    ----------
    video_path : str
        Absolute path to the dashcam .ts video file.
    carstate_csv : str
        Absolute path to the matching busCAN CSV file that must contain at
        least the columns:
            - 't'              : absolute timestamp in nanoseconds
            - 'steeringTorque' : steering torque value (raw sensor units)
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing {base_name}")


    
    # STEP 1 — Load and normalize busCAN telemetry
    
    # Read the CSV and validate that the required columns are present.
    carstate = pd.read_csv(carstate_csv)
    if "t" not in carstate.columns or "steeringTorque" not in carstate.columns:
        print(f"  ERROR: CSV {carstate_csv} is missing required columns ('t', 'steeringTorque').")
        return

    # Convert the absolute nanosecond timestamp to relative seconds so it can
    # be compared directly with video frame timestamps (which also start at 0).
    carstate["timestamp_sec"] = (carstate["t"] - carstate["t"].iloc[0]) / 1e9
    carstate = carstate.sort_values("timestamp_sec")


    
    # STEP 2 — Extract frames from the dashcam video
    
    cap = cv2.VideoCapture(video_path)

    # Attempt to read the encoded FPS from the video container metadata.
    # Some .ts recordings do not expose this reliably, so we fall back to the
    # configured default.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if np.isnan(fps) or fps <= 0:
        fps = fps_default
        print(f"  FPS not detected in metadata — using default: {fps_default}")
    else:
        print(f"  Detected FPS: {fps:.2f}")

    # Create the output sub-directory for this video segment
    out_dir = os.path.join(output_root, base_name)
    os.makedirs(out_dir, exist_ok=True)

    frame_idx        = 0
    video_timestamps = []   # relative timestamp (seconds) of each frame
    filenames        = []   # saved image filename for each frame

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video stream
            break

        # Compute the relative timestamp of this frame assuming constant FPS
        ts = frame_idx / fps

        # Save frame as a zero-padded JPEG (e.g. 00042.jpg)
        filename = f"{frame_idx:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, filename), frame)

        video_timestamps.append(ts)
        filenames.append(filename)
        frame_idx += 1

    cap.release()
    print(f"   {frame_idx} frames extracted.")

    # Build a DataFrame that maps each saved image to its relative timestamp
    df_video = pd.DataFrame({"timestamp_sec": video_timestamps, "filename": filenames})

    
    # STEP 3 — Synchronize frames with busCAN data (temporal alignment)
    
    # merge_asof performs a nearest-neighbor join on the sorted timestamp
    # column.  The 'tolerance' parameter ensures that only telemetry samples
    # within `tolerance_sec` seconds of the frame timestamp are matched;
    # frames outside this window receive NaN for steeringTorque and are later
    # discarded to avoid using stale sensor readings as ground truth.
    df_sync = pd.merge_asof(
        df_video.sort_values("timestamp_sec"),
        carstate[["timestamp_sec", "steeringTorque"]],
        on="timestamp_sec",
        direction="nearest",
        tolerance=tolerance_sec
    )

    # Quality check: warn if less than half of frames could be matched.
    # This usually indicates a timestamp mismatch between the video and the CSV
    # (e.g. different recording sessions were paired by mistake).
    porcentaje_valido = df_sync["steeringTorque"].notna().mean() * 100
    if porcentaje_valido < 50:
        print(f"  WARNING: only {porcentaje_valido:.1f}% of frames matched a torque sample.")
        print(f"  Verify that the video and the CSV belong to the same recording session.")

    
    # STEP 4 — Write the annotation file (data.txt)
    
    # Each line in data.txt follows the format expected by the preprocessing
    # and training scripts:
    #
    #     <filename> <steeringTorque>
    #
    # Only frames that obtained a valid torque match are included.
    data_txt_path = os.path.join(out_dir, "data.txt")
    with open(data_txt_path, "w") as f:
        for _, row in df_sync.iterrows():
            if pd.notna(row["steeringTorque"]):
                f.write(f"{row['filename']} {row['steeringTorque']}\n")

    frames_con_torque = df_sync["steeringTorque"].notna().sum()
    frames_sin_torque = df_sync["steeringTorque"].isna().sum()
    print(f"   Labeled frames  : {frames_con_torque}")
    print(f"   Discarded frames: {frames_sin_torque}  (no torque match within tolerance)")
    print(f"  Dataset ready at  : {out_dir}\n")


# MAIN LOOP — iterate over all video segments in the videos directory

for file_name in os.listdir(videos_dir):
    # Only process dashcam video files (.ts extension)
    if not file_name.endswith(".ts"):
        continue

    # Only process the camera stream files (qcamera), not other .ts streams
    # that may exist in the same directory (e.g. fcamera, dcamera).
    if "--qcamera.ts" not in file_name:
        continue

    video_path = os.path.join(videos_dir, file_name)

    # Derive the common segment identifier by stripping the camera-stream suffix.
    # Example: "2024-01-15--10-30-00--0--qcamera.ts"
    #         "2024-01-15--10-30-00--0--"
    base = file_name.replace("--qcamera.ts", "")

    # busCAN CSVs may have been exported with slightly different naming
    # conventions depending on the log type (rlog vs qlog) and parser version.
    # We try all known variants and use the first one that exists on disk.
    posibles_csv = [
        base + "--rlog_carState.csv",
        base + "--rlog._carState.csv",
        base + "--qlog_carState.csv",
        base + "--qlog._carState.csv",
    ]

    carstate_csv = None
    for nombre_csv in posibles_csv:
        ruta_csv = os.path.join(csvs_dir, nombre_csv)
        if os.path.exists(ruta_csv):
            carstate_csv = ruta_csv
            break

    if carstate_csv is None:
        print(f"No matching CSV found for {file_name} — skipping.")
        continue

    procesar_video(video_path, carstate_csv)


print("Synchronization completed for all video segments.")