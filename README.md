ANAID – PilotNet Training & Dataset Preparation
Supporting Material for the Paper:
“ANAID: Autonomous Naturalistic obstacle‑Avoidance Interaction Dataset”
Published in MDPI – DATA
This repository provides example code and preprocessing utilities for working with the ANAID dataset and training a PilotNet model for end‑to‑end autonomous driving. It accompanies the public Kaggle release of the dataset and the associated academic publication.

📦 Dataset
The ANAID: Autonomous Naturalistic obstacle‑Avoidance Interaction Dataset is available on Kaggle:
🔗 Kaggle Dataset
https://www.kaggle.com/datasets/mgarciafernandez/anaid-autonomous-naturalistic-driving
📘 DOI
10.34740/kaggle/dsv/15045737
The dataset contains synchronized ego‑vehicle video, obstacle‑interaction events, control signals, and additional metadata essential for naturalistic driving and obstacle‑avoidance research.

📁 Repository Structure
.
├── pilotnet_example.py
├── prepare_pilotnet_database.py
└── README.md

pilotnet_example.py
This script demonstrates:

Loading the processed ANAID dataset
Training a PilotNet convolutional neural network
Integration in a Jupyter Notebook workflow
Preprocessing, batching, and evaluation procedures

It serves as reference code to reproduce the experiments described in the ANAID paper.

prepare_pilotnet_database.py
This script implements the data‑reduction and preprocessing pipeline, transforming the raw ANAID dataset into a value‑added, PilotNet‑ready training set.
Key steps include:

Frame extraction
Annotation alignment
Obstacle/interaction filtering
Resolution and format normalization
Steering/control signal preparation
Dataset structuring for efficient deep learning workflows

The resulting dataset is directly usable by pilotnet_example.py.

🚀 Getting Started
1. Download the ANAID dataset
Shellkaggle datasets download -d mgarciafernandez/anaid-autonomous-naturalistic-drivingunzip anaid-autonomous-naturalistic-driving.zipMostrar más líneas
2. Prepare the PilotNet‑ready dataset
Shellpython prepare_pilotnet_database.py \    --input <path_to_raw_dataset> \    --output <path_to_processed_dataset>``Mostrar más líneas
3. Train a PilotNet model with ANAID
Shellpython pilotnet_example.py \    --data <path_to_processed_dataset>Mostrar más líneas

🧠 About PilotNet
PilotNet is the CNN architecture introduced by NVIDIA for end‑to‑end steering prediction from camera inputs.
This repository adapts it for naturalistic obstacle‑avoidance interaction behavior, leveraging the unique conditions captured in ANAID.

📄 Citation
If you use this repository or dataset, please cite:
Paper
Garcia‑Fernandez, M.; et al.
“ANAID: Autonomous Naturalistic obstacle‑Avoidance Interaction Dataset.”
MDPI – DATA, 2026.
Dataset
Garcia-Fernandez, M. (2026). ANAID Autonomous Naturalistic Driving [Data set].
Kaggle. https://doi.org/10.34740/kaggle/dsv/15045737


🤝 Contributing
Contributions, improvements to the preprocessing pipeline, and extensions to the PilotNet example are welcome.
Please open an issue or submit a pull request.

📧 Contact
For questions related to the dataset, publication, or reproduction of results:
Manuel García Fernández
