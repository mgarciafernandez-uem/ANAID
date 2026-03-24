"""
prepare_pilotnet_dataset.py

Preprocessing pipeline for the PilotNet dataset.

This script:
1. Browse through the run1, run2, run3, and run4 folders within the raw dataset.
2. Read the image-torque mappings from telemetry_data/frame-torque.txt.
3. Verify that the images exist and can be opened correctly.
4. Calculate global torque statistics.
5. Normalize the torques to the range [-1, 1].
6. Crop the images to the input size used by PilotNet.
7. Convert the images to tensors.
8. Save the processed dataset and a configuration file containing the statistics.

Dataset structure:

dataset_root/
│
├── run1/
│   ├── telemetry_data/
│   │   └── frame-torque.txt
│   └── video_data/
│       └── frame_videos/
│           ├── frame_0001.jpg
│           ├── frame_0002.jpg
│           └── ...
│
├── run2/
├── run3/
└── run4/
"""

import os
import json
import argparse
from typing import List, Dict, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm


# IMAGE PREPROCESSING CONFIGURATION

# Transformation used in your work:
# - CenterCrop((66, 200)): adapts the image to the format expected by PilotNet
# - ToTensor(): converts the PIL image to a PyTorch tensor
IMAGE_TRANSFORM = transforms.Compose([
    transforms.CenterCrop((66, 200)),
    transforms.ToTensor(),
])


# AUXILIARY FUNCTIONS

def parse_arguments() -> argparse.Namespace:
    """
    Defines and processes command-line arguments.

    Returns
    ------
    argparse.Namespace
        Object with the parameters entered by the user.
    """
    parser = argparse.ArgumentParser(
        description="Preprocessing the dataset for PilotNet training"
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the root folder of the raw dataset."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the processed dataset will be saved."
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        default=["run1", "run2", "run3", "run4"],
        help="List of runs to process. Example: --runs run1 run2 run3 run4"
    )

    return parser.parse_args()


def load_image_torque_pairs(
    dataset_root: str,
    run_folders: List[str]
) -> Tuple[List[str], List[float], Dict[str, int]]:
    """
    Traverses the specified folders and loads the image paths along with their torques.
    
    Parameters
    ----------
    dataset_root : str
        Root folder of the raw dataset.
    run_folders : List[str]
        List of runs to process.

    Returns
    -------
    Tuple[List[str], List[float], Dict[str, int]]
        - List of absolute paths to valid images.
        - List of associated torques.
        - Dictionary with the number of samples found per run.
    """
    all_images = []
    all_torques = []
    samples_per_run = {}

    print("\n[1/5] Loading image paths and torques")

    for run_name in run_folders:
        run_path = os.path.join(dataset_root, run_name)

        if not os.path.isdir(run_path):
            print(f"    Folder not found, skipping: {run_path}")
            continue

        telemetry_dir = os.path.join(run_path, "telemetry_data")
        video_dir = os.path.join(run_path, "video_data", "frame_videos")
        frame_torque_file = os.path.join(telemetry_dir, "frame-torque.txt")

        if not os.path.exists(frame_torque_file):
            print(f"     frame-torque.txt not found in {telemetry_dir}")
            continue

        if not os.path.isdir(video_dir):
            print(f"     Image folder not found in {video_dir}")
            continue

        count_before = len(all_images)

        with open(frame_torque_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                # Each line must have exactly:
                # image_name torque
                if len(parts) != 2:
                    continue

                img_name, torque_str = parts
                img_path = os.path.join(video_dir, img_name)

                if os.path.exists(img_path):
                    all_images.append(img_path)
                    all_torques.append(float(torque_str))

        count_after = len(all_images)
        samples_per_run[run_name] = count_after - count_before
        print(f"     {run_name}: {samples_per_run[run_name]} samples")

    print(f"\n     Total samples found: {len(all_images)}")
    return all_images, all_torques, samples_per_run


def compute_torque_statistics(all_torques: List[float], all_images: List[str]) -> Dict:
    """
    Computes global statistics on the original torques.

    Parameters
    ----------
    all_torques : List[float]
        List of unnormalized torques.
    all_images : List[str]
        List of image paths associated with each torque.

    Returns
    -------
    Dict
        Dictionary with dataset statistics.
    """
    print("\n[2/5] Analyzing torque distribution")

    torques_tensor = torch.tensor(all_torques, dtype=torch.float32)

    min_torque = torques_tensor.min().item()
    max_torque = torques_tensor.max().item()
    mean_torque = torques_tensor.mean().item()
    std_torque = torques_tensor.std().item()

    idx_min = torch.argmin(torques_tensor).item()
    idx_max = torch.argmax(torques_tensor).item()

    num_positive = (torques_tensor > 0).sum().item()
    num_negative = (torques_tensor < 0).sum().item()
    num_zero = (torques_tensor == 0).sum().item()

    print(f"    Min:   {min_torque:.4f}")
    print(f"    Max:   {max_torque:.4f}")
    print(f"    Mean:  {mean_torque:.4f}")
    print(f"    Std:   {std_torque:.4f}")
    print(f"    Positive: {num_positive}")
    print(f"    Negative: {num_negative}")
    print(f"    Zeros:     {num_zero}")
    print(f"    Image with minimum torque: {all_images[idx_min]}")
    print(f"    Image with maximum torque: {all_images[idx_max]}")

    stats = {
        "min": min_torque,
        "max": max_torque,
        "mean": mean_torque,
        "std": std_torque,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_zero": num_zero,
        "min_image_path": all_images[idx_min],
        "max_image_path": all_images[idx_max],
    }

    return stats


def normalize_torques(all_torques: List[float], stats: Dict) -> Tuple[torch.Tensor, Dict]:
    """
    Normalizes original torques to the range [-1, 1] using symmetric Min-Max normalization.

    Formula:
        norm = 2 * (x - min) / (max - min) - 1

    Parameters
    ----------
    all_torques : List[float]
        List of original torques.
    stats : Dict
        Previously calculated statistics.

    Returns
    -------
    Tuple[torch.Tensor, Dict]
        - Tensor with normalized torques.
        - Dictionary with normalization parameters.
    """
    print("\n[3/5] Normalizing torques")

    torques_tensor = torch.tensor(all_torques, dtype=torch.float32)
    min_torque = stats["min"]
    max_torque = stats["max"]
    torque_range = max_torque - min_torque

    if torque_range == 0:
        raise ValueError("Cannot normalize: all torques are equal.")

    torques_normalized = 2 * (torques_tensor - min_torque) / torque_range - 1

    normalization_params = {
        "method": "minmax_symmetric",
        "min_torque": min_torque,
        "max_torque": max_torque,
        "range": torque_range,
    }

    print(f"    Original range: [{min_torque:.4f}, {max_torque:.4f}]")
    print(
        f"    Normalized range: "
        f"[{torques_normalized.min().item():.4f}, {torques_normalized.max().item():.4f}]"
    )

    return torques_normalized, normalization_params


def process_images(
    image_paths: List[str],
    torques_normalized: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Opens, crops, and converts images to tensors.

    Only samples whose images can be loaded correctly are saved.
    If an image is corrupted or cannot be opened, it is discarded along with its torque.

    Parameters
    ----------
    image_paths : List[str]
        List of image paths.
    torques_normalized : torch.Tensor
        Tensor with already normalized torques.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, int]
        - Tensor of processed images.
        - Tensor of valid torques.
        - Number of discarded images.
    """
    print("\n[4/5] Processing images")

    images_tensor = []
    valid_torques = []
    discarded_images = 0

    for i, img_path in enumerate(tqdm(image_paths, desc="    Processing")):
        try:
            img = Image.open(img_path).convert("RGB")
            img = IMAGE_TRANSFORM(img)

            images_tensor.append(img)
            valid_torques.append(torques_normalized[i])

        except (UnidentifiedImageError, OSError):
            print(f"\n     Corrupted or unreadable image, discarding: {img_path}")
            discarded_images += 1
            continue

    if len(images_tensor) == 0:
        raise RuntimeError("No valid images could be processed.")

    images_tensor = torch.stack(images_tensor)
    valid_torques = torch.stack(valid_torques)

    print(f"     Valid images: {len(images_tensor)}")
    print(f"     Discarded images: {discarded_images}")
    print(f"     Shape of image tensor: {tuple(images_tensor.shape)}")

    return images_tensor, valid_torques, discarded_images


def save_outputs(
    output_dir: str,
    images_tensor: torch.Tensor,
    torques_tensor: torch.Tensor,
    stats: Dict,
    normalization_params: Dict,
    run_folders: List[str],
    samples_per_run: Dict[str, int],
    num_original_samples: int,
    num_valid_samples: int,
    num_discarded_samples: int
) -> None:
    """
    Saves the processed dataset and preprocessing metadata.

    Two files are generated:
    - processed_data.pt : tensors and information for training
    - preprocessing_info.json : readable and reusable metadata

    Parameters
    ----------
    output_dir : str
        Output folder.
    images_tensor : torch.Tensor
        Final image tensor.
    torques_tensor : torch.Tensor
        Final tensor of normalized torques.
    stats : Dict
        Original statistics of torques.
    normalization_params : Dict
        Parameters of the applied normalization.
    run_folders : List[str]
        Runs requested by the user.
    samples_per_run : Dict[str, int]
        Number of samples per run.
    num_original_samples : int
        Samples detected before filtering corrupt images.
    num_valid_samples : int
        Final valid samples.
    num_discarded_samples : int
        Samples discarded due to read errors.
    """
    print("\n[5/5] Saving processed data")

    os.makedirs(output_dir, exist_ok=True)

    # Main file for PyTorch
    processed_data = {
        "images": images_tensor,
        "torques": torques_tensor,
        "normalization_params": normalization_params,
        "original_stats": stats,
        "num_samples_original": num_original_samples,
        "num_samples_valid": num_valid_samples,
        "num_samples_discarded": num_discarded_samples,
        "runs_included": run_folders,
        "samples_per_run": samples_per_run,
    }

    output_pt = os.path.join(output_dir, "processed_data.pt")
    torch.save(processed_data, output_pt)

    # Readable JSON file for quick inspection and documentation
    output_json = os.path.join(output_dir, "preprocessing_info.json")
    info_json = {
        "runs_included": run_folders,
        "samples_per_run": samples_per_run,
        "num_samples_original": num_original_samples,
        "num_samples_valid": num_valid_samples,
        "num_samples_discarded": num_discarded_samples,
        "image_tensor_shape": list(images_tensor.shape),
        "original_stats": stats,
        "normalization": {
            **normalization_params,
            "normalized_min": float(torques_tensor.min().item()),
            "normalized_max": float(torques_tensor.max().item()),
            "formula_normalization": (
                "norm = 2 * (real - min_torque) / (max_torque - min_torque) - 1"
            ),
            "formula_denormalization": (
                "real = (norm + 1) * (max_torque - min_torque) / 2 + min_torque"
            ),
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(info_json, f, indent=4, ensure_ascii=False)

    size_mb = os.path.getsize(output_pt) / (1024 ** 2)

    print(f"     Processed dataset saved to: {output_pt}")
    print(f"     Metadata saved to:        {output_json}")
    print(f"     Size of .pt file:        {size_mb:.2f} MB")


# MAIN FUNCTION

def main() -> None:
    """
    Executes the complete preprocessing pipeline.
    """
    args = parse_arguments()

    print("=" * 70)
    print("PILOTNET DATASET PREPROCESSING")
    print("=" * 70)
    print(f"Root dataset: {args.dataset_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Runs to process: {', '.join(args.runs)}")

    # 1. Load image-torque pairs
    all_images, all_torques, samples_per_run = load_image_torque_pairs(
        dataset_root=args.dataset_root,
        run_folders=args.runs
    )

    if len(all_images) == 0:
        raise RuntimeError(
            "No valid images found. "
            "Check the paths and dataset structure."
        )

    # 2. Calculate global statistics
    stats = compute_torque_statistics(all_torques, all_images)

    # 3. Normalize torques
    torques_normalized, normalization_params = normalize_torques(all_torques, stats)

    # 4. Process images
    images_tensor, valid_torques, discarded_images = process_images(
        image_paths=all_images,
        torques_normalized=torques_normalized
    )

    # 5. Save outputs
    save_outputs(
        output_dir=args.output_dir,
        images_tensor=images_tensor,
        torques_tensor=valid_torques,
        stats=stats,
        normalization_params=normalization_params,
        run_folders=args.runs,
        samples_per_run=samples_per_run,
        num_original_samples=len(all_images),
        num_valid_samples=len(images_tensor),
        num_discarded_samples=discarded_images
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)
    print(f"Processed runs: {', '.join(args.runs)}")
    print(f"Original samples: {len(all_images)}")
    print(f"Valid samples:    {len(images_tensor)}")
    print(f"Discarded samples:{discarded_images}")
    print("=" * 70)


if __name__ == "__main__":
    main()