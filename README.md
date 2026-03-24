
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
├── pilotnet_example.py
├── prepare_pilotnet_database.py
└── README.md
```

### `pilotnet_example.py`
An end‑to‑end example showing how to:

- Load the processed ANAID value‑added dataset  
- Train a **PilotNet** model  
- Use the workflow inside a Jupyter Notebook  
- Perform data batching, preprocessing, and evaluation  

This script serves as reference code to reproduce experiments from the ANAID paper.

---

### `prepare_pilotnet_database.py`
Implements the full **data‑reduction and preprocessing pipeline** needed to convert the **raw ANAID dataset** into a **PilotNet‑ready dataset**.

The pipeline includes:

- Frame extraction  
- Annotation synchronization  
- Filtering and reduction  
- Resolution normalization  
- Steering/control signal formatting  
- Dataset structuring for deep learning workflows  

The resulting dataset is used directly by `pilotnet_example.py`.

---

## 🚀 Getting Started

### 1. Download the ANAID dataset

```bash
kaggle datasets download -d mgarciafernandez/anaid-autonomous-naturalistic-driving
unzip anaid-autonomous-naturalistic-driving.zip
```

### 2. Prepare the PilotNet‑ready dataset

```bash
python prepare_pilotnet_database.py     --input <path_to_raw_dataset>     --output <path_to_processed_dataset>
```

### 3. Train PilotNet using ANAID

```bash
python pilotnet_example.py     --data <path_to_processed_dataset>
```

---

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
