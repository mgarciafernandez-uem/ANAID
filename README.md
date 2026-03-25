
# ANAID – PilotNet Training & Dataset Preparation
**Supporting Material for the Paper:**  
**_“ANAID: Autonomous Naturalistic obstacle‑Avoidance Interaction Dataset”_**  
Published in **MDPI – DATA**

This repository provides example code and preprocessing utilities for working with the **ANAID** dataset and for training a **PilotNet** model for end‑to‑end autonomous driving. It accompanies the Kaggle dataset release and the associated academic publication.

---

## 📦 Dataset

The **ANAID: Autonomous Naturalistic obstacle‑Avoidance Interaction Dataset** is publicly available:

- **Kaggle:** https://www.kaggle.com/datasets/mgarciafernandez/anaid-autonomous-naturalistic-driving  
- **DOI:** `10.34740/kaggle/dsv/15045737`

The dataset includes ego‑vehicle video, obstacle‑interaction events, control commands, and metadata relevant to naturalistic obstacle‑avoidance driving.

---

## 📁 Repository Structure

```
.
├── pilotnet_example.ipynb
├── prepare_pilotnet_database.py
├── car_turn_analysis.py
├── car_turn_analysis_videos.py
├── sync_video_frames_with_telemetry_global_dataset.py
├── sync_videos_with_busCANData_pilotnet.py
└── README.md
```

### `pilotnet_example.ipynb`
An end‑to‑end example showing how to:

- Load the processed ANAID value‑added dataset  
- Train a **PilotNet** model  
- Use the workflow inside a Jupyter Notebook  
- Perform data batching, preprocessing, and evaluation  

This script serves as reference code to reproduce experiments from the ANAID paper.

---

### `car_turn_analysis_videos.py`
Video Fragment Extraction Pipeline for Driving Behavior Analysis

This script extracts video clips corresponding to:
1. Evasion maneuvers ("evasions") detected in a CSV file.
2. Normal driving segments, defined as the complementary intervals
   not covered by any evasion segment.

Detections from both methods are merged, deduplicated, and filtered to remove
common false positives (e.g. "pre-mountain" artefacts caused by the signal
rising into a much larger maneuver).

### `car_turn_analysis.py`
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

### `sync_video_frames_with_telemetry_global_dataset.py`
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



### `sync_videos_with_busCANData_pilotnet.py`
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

### `prepare_pilotnet_database.py`
Implements the full **data‑reduction and preprocessing pipeline** needed to convert the **raw ANAID dataset** into a **PilotNet‑ready dataset**.

The pipeline includes:

- Frame extraction  
- Annotation synchronization  
- Filtering and reduction  
- Resolution normalization  
- Steering/control signal formatting  
- Dataset structuring for deep learning workflows  

The resulting dataset is used directly by `pilotnet_example.ipynb`.


## 🧠 About PilotNet

PilotNet is a CNN architecture introduced by NVIDIA for end‑to‑end autonomous steering prediction.  
This repository adapts it for naturalistic **obstacle‑avoidance behavior** captured in the ANAID dataset.

---

## 📄 Citation

If you use this repository or dataset, please cite:

### Paper
Garcia‑Fernandez, M.; *et al.*  
**“ANAID: Autonomous Naturalistic obstacle‑Avoidance Interaction Dataset.”**  
MDPI – DATA, 2026.

### Dataset
```
Garcia-Fernandez, M. (2026). ANAID Autonomous Naturalistic Driving [Data set].
Kaggle. https://doi.org/10.34740/kaggle/dsv/15045737
```

---

## 🤝 Contributing

Contributions and improvements to the data pipeline or the PilotNet example are welcome.  
Feel free to open an issue or submit a pull request.

---

## 📧 Contact

For questions about the dataset or publication:

**Manuel García Fernández**  
