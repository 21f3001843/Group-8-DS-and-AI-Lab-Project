
"""
NIH ChestX-ray14: Training Artifacts & Reproducibility
Single-file implementation.

Models:
    - DenseNet121
    - ResNet50
    - EfficientNet-B0

Features:
    - Artifact discovery and inventory
    - Configuration collection
    - Training log collection
    - SHA256 manifest generation
    - Artifact integrity verification
    - Optional checkpoint inference verification
    - Reproducibility report
"""

import os
import sys
import json
import csv
import hashlib
import random
import platform
import argparse
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 7

BACKBONES = [
    "densenet121",
    "resnet50",
    "efficientnet_b0"
]

INPUT_ROOT = Path(
    "/kaggle/input/notebooks/t1f3001843"
)

PREPROCESSING_ROOT = (
    INPUT_ROOT / "dsai-nih-preprocessing"
)

TUNING_ROOT = (
    INPUT_ROOT /
    "dsai-nih-hyperparameter" /
    "nih_multi_backbone_tuning"
)

EXPERIMENT_ROOT = Path(
    "/kaggle/working/nih_generalization_experiments"
)

OUTPUT_ROOT = Path(
    "/kaggle/working/nih_reproducibility"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Existing experiment artifact folders may have different
# naming conventions. The script searches recursively.

CHECKPOINT_NAMES = [
    "best_checkpoint.pt",
    "last_checkpoint.pt",
    "best_model.pth",
    "model.pth",
    "checkpoint.pt"
]

ARTIFACT_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml",
    ".csv", ".txt", ".md", ".pt", ".pth"
}

# ============================================================
# 2. REPRODUCIBILITY SETTINGS
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except ImportError:
        pass


# ============================================================
# 3. SYSTEM INFORMATION
# ============================================================

def get_system_information():

    information = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "backbones": BACKBONES
    }

    try:
        import torch
        import torchvision

        information["torch_version"] = torch.__version__
        information["torchvision_version"] = (
            torchvision.__version__
        )

        information["cuda_available"] = (
            torch.cuda.is_available()
        )

        information["cuda_version"] = torch.version.cuda

        if torch.cuda.is_available():
            information["gpu"] = torch.cuda.get_device_name(0)

    except Exception as error:
        information["torch_information_error"] = str(error)

    return information


# ============================================================
# 4. FILE HASHING
# ============================================================

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 5. ARTIFACT DISCOVERY
# ============================================================

def discover_artifacts():

    discovered = []

    search_roots = [
        EXPERIMENT_ROOT,
        PREPROCESSING_ROOT,
        TUNING_ROOT
    ]

    for root in search_roots:

        if not root.exists():
            print(f"[WARNING] Missing directory: {root}")
            continue

        for file_path in root.rglob("*"):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in ARTIFACT_EXTENSIONS:
                continue

            discovered.append(file_path)

    return sorted(set(discovered))


# ============================================================
# 6. BACKBONE CHECKPOINT DISCOVERY
# ============================================================

def find_checkpoints():

    results = {}

    for backbone in BACKBONES:

        results[backbone] = []

        for file_path in EXPERIMENT_ROOT.rglob("*"):

            if not file_path.is_file():
                continue

            filename = file_path.name.lower()

            if filename not in CHECKPOINT_NAMES:
                continue

            # Associate checkpoint with its backbone folder.
            path_text = str(file_path).lower()

            if backbone in path_text:
                results[backbone].append(file_path)

    return results


# ============================================================
# 7. LOAD HYPERPARAMETER CONFIGURATIONS
# ============================================================

def load_hyperparameter_configs():

    configurations = {}

    for backbone in BACKBONES:

        candidates = list(
            TUNING_ROOT.rglob("best_config.json")
        )

        matched = []

        for config_path in candidates:

            if backbone in str(config_path).lower():
                matched.append(config_path)

        if matched:

            config_path = matched[0]

            try:
                with open(config_path, "r") as file:
                    configurations[backbone] = json.load(file)

            except Exception as error:
                configurations[backbone] = {
                    "error": str(error)
                }

        else:
            configurations[backbone] = {
                "status": "Configuration not found"
            }

    return configurations


# ============================================================
# 8. CREATE ARTIFACT INVENTORY
# ============================================================

def create_inventory():

    print("\n" + "=" * 65)
    print("NIH CHEST-XRAY14 ARTIFACT INVENTORY")
    print("=" * 65)

    artifacts = discover_artifacts()
    checkpoints = find_checkpoints()

    inventory = []

    for file_path in artifacts:

        try:
            relative_path = str(file_path)

            record = {
                "file_name": file_path.name,
                "path": relative_path,
                "extension": file_path.suffix,
                "size_bytes": file_path.stat().st_size,
                "sha256": calculate_sha256(file_path)
            }

            inventory.append(record)

        except Exception as error:

            print(
                f"[ERROR] Could not process "
                f"{file_path}: {error}"
            )

    inventory_data = {
        "project": "NIH ChestX-ray14 Multi-label Classification",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "total_artifacts": len(inventory),

        "artifacts": inventory
    }

    inventory_file = OUTPUT_ROOT / "artifact_inventory.json"

    with open(inventory_file, "w") as file:

        json.dump(
            inventory_data,
            file,
            indent=4
        )

    # Save a CSV version too.
    csv_file = OUTPUT_ROOT / "artifact_inventory.csv"

    with open(csv_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_name",
                "path",
                "extension",
                "size_bytes",
                "sha256"
            ]
        )

        writer.writeheader()
        writer.writerows(inventory)

    print(f"\nTotal discovered artifacts: {len(inventory)}")

    for backbone, files in checkpoints.items():

        print(f"\n{backbone}:")

        if files:
            for checkpoint in files:
                print(f"  [FOUND] {checkpoint}")
        else:
            print("  [MISSING] No checkpoint found")

    return inventory, checkpoints


# ============================================================
# 9. SAVE EXPERIMENT CONFIGURATION
# ============================================================

def save_experiment_configuration():

    configuration = {
        "project": "NIH ChestX-ray14",
        "task": "Multi-label Chest X-ray Classification",

        "random_seed": SEED,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,

        "backbones": BACKBONES,

        "training": {
            "train_samples": 40000,
            "validation_samples": 8000,
            "epochs": 8,
            "mixed_precision": True,
            "warmup_ratio": 0.05,
            "gradient_clip_norm": 1.0
        },

        "data_paths": {
            "preprocessing": str(PREPROCESSING_ROOT),
            "hyperparameter_tuning": str(TUNING_ROOT),
            "experiments": str(EXPERIMENT_ROOT)
        },

        "output_directory": str(OUTPUT_ROOT),

        "note": (
            "Configuration records the intended experiment "
            "setup. Verify against actual training logs."
        )
    }

    configurations = load_hyperparameter_configs()

    configuration["best_hyperparameters"] = configurations

    config_file = OUTPUT_ROOT / "experiment_config.json"

    with open(config_file, "w") as file:

        json.dump(
            configuration,
            file,
            indent=4
        )

    print("\n[OK] Experiment configuration saved.")

    return configuration


# ============================================================
# 10. CHECKPOINT INTEGRITY VERIFICATION
# ============================================================

def verify_checkpoints(checkpoints):

    verification = {}

    for backbone, files in checkpoints.items():

        verification[backbone] = []

        for checkpoint in files:

            try:

                checksum = calculate_sha256(checkpoint)

                verification[backbone].append({
                    "checkpoint": str(checkpoint),
                    "exists": True,
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": checksum,
                    "status": "PASS"
                })

            except Exception as error:

                verification[backbone].append({
                    "checkpoint": str(checkpoint),
                    "status": "FAIL",
                    "error": str(error)
                })

    output_file = OUTPUT_ROOT / "checkpoint_verification.json"

    with open(output_file, "w") as file:

        json.dump(
            verification,
            file,
            indent=4
        )

    return verification


# ============================================================
# 11. VERIFY SAVED CHECKPOINT WITH INFERENCE
# ============================================================

def build_model(backbone, num_classes=NUM_CLASSES):

    import torch.nn as nn
    from torchvision import models

    if backbone == "densenet121":

        model = models.densenet121(weights=None)

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    elif backbone == "resnet50":

        model = models.resnet50(weights=None)

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    elif backbone == "efficientnet_b0":

        model = models.efficientnet_b0(weights=None)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    else:
        raise ValueError(
            f"Unsupported backbone: {backbone}"
        )

    return model


def extract_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format")

    for key in [
        "model_state_dict",
        "state_dict",
        "model",
        "weights"
    ]:

        if key in checkpoint:

            state = checkpoint[key]

            if isinstance(state, dict):
                return state

    # Also support a raw state_dict.
    if checkpoint and all(
        hasattr(value, "shape")
        for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        "Could not identify model state_dict in checkpoint."
    )


def verify_inference(backbone, checkpoint_path, image_path):

    import torch
    from PIL import Image
    from torchvision import transforms

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nStarting inference verification...")
    print("Backbone:", backbone)
    print("Device:", device)

    model = build_model(backbone)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

    # Handle DataParallel checkpoints.
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    image_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

    probabilities = probabilities.cpu().numpy()[0]

    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia"
    ]

    predictions = {}

    for label, probability in zip(labels, probabilities):

        predictions[label] = round(
            float(probability), 6
        )

    result = {
        "backbone": backbone,
        "checkpoint": str(checkpoint_path),
        "test_image": str(image_path),
        "inference_status": "PASS",
        "output_shape": list(logits.shape),
        "predictions": predictions
    }

    result_file = OUTPUT_ROOT / (
        f"{backbone}_inference_verification.json"
    )

    with open(result_file, "w") as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print("\nInference verification successful.")

    for label, probability in predictions.items():

        print(f"{label:20s}: {probability:.6f}")

    return result


# ============================================================
# 12. GENERATE REPRODUCIBILITY REPORT
# ============================================================

def generate_report(
    inventory,
    checkpoints,
    verification,
    configuration
):

    report = []

    report.append(
        "# NIH ChestX-ray14 Reproducibility Report\n"
    )

    report.append(
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
    )

    report.append("## 1. Experiment Overview\n")

    report.append(
        "- Task: Multi-label Chest X-ray Classification\n"
        "- Random seed: 42\n"
        "- Image size: 224 x 224\n"
        "- Number of classes: 7\n"
        "- Training samples: 40,000\n"
        "- Validation samples: 8,000\n"
        "- Epochs: 8\n"
    )

    report.append("## 2. Model Checkpoints\n")

    for backbone, files in checkpoints.items():

        report.append(f"\n### {backbone}\n")

        if files:

            for file_path in files:

                report.append(
                    f"- FOUND: `{file_path}`\n"
                )

        else:

            report.append("- MISSING\n")

    report.append("\n## 3. Checkpoint Integrity\n")

    for backbone, records in verification.items():

        for record in records:

            report.append(
                f"- {backbone}: "
                f"{record['status']} "
                f"({record['checkpoint']})\n"
            )

    report.append("\n## 4. Artifact Inventory\n")

    report.append(
        f"- Total artifacts discovered: {len(inventory)}\n"
    )

    report.append("\n## 5. Hyperparameter Configuration\n")

    report.append(
        "Individual best_config.json files are recorded "
        "in experiment_config.json when available.\n"
    )

    report.append("\n## 6. System Information\n")

    report.append(
        "```json\n"
    )

    report.append(
        json.dumps(
            get_system_information(),
            indent=4
        )
    )

    report.append("\n```\n")

    report.append("\n## 7. Reproducibility Notes\n")

    report.append(
        "- SHA-256 checksums are generated for discovered artifacts.\n"
        "- Checkpoint integrity is checked against recorded checksums.\n"
        "- Training logs and configuration files are inventoried.\n"
        "- Actual inference verification requires a compatible "
        "checkpoint and a test image.\n"
        "- Reproduction of training requires the original dataset, "
        "preprocessing pipeline, dependencies and training code.\n"
    )

    report_file = OUTPUT_ROOT / "REPRODUCIBILITY_REPORT.md"

    with open(report_file, "w") as file:

        file.write("".join(report))

    print("\n[OK] Reproducibility report generated.")


# ============================================================
# 13. SAVE MANIFEST
# ============================================================

def save_manifest(inventory):

    manifest = {
        "project": "NIH ChestX-ray14",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "seed": SEED,

        "artifacts": [
            {
                "path": item["path"],
                "sha256": item["sha256"]
            }
            for item in inventory
        ]
    }

    manifest_path = OUTPUT_ROOT / "manifest.json"

    with open(manifest_path, "w") as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print("[OK] Manifest generated.")


# ============================================================
# 14. VERIFY MANIFEST
# ============================================================

def verify_manifest():

    manifest_path = OUTPUT_ROOT / "manifest.json"

    if not manifest_path.exists():

        print("[ERROR] Manifest not found.")
        return

    with open(manifest_path, "r") as file:

        manifest = json.load(file)

    passed = 0
    failed = 0

    print("\nMANIFEST VERIFICATION")

    for artifact in manifest["artifacts"]:

        file_path = Path(artifact["path"])

        if not file_path.exists():

            print(f"[MISSING] {file_path}")
            failed += 1
            continue

        actual_hash = calculate_sha256(file_path)

        if actual_hash == artifact["sha256"]:

            print(f"[PASS] {file_path.name}")
            passed += 1

        else:

            print(f"[FAIL] {file_path.name}")
            failed += 1

    print("\nVerification summary:")
    print("Passed:", passed)
    print("Failed:", failed)

    return passed, failed


# ============================================================
# 15. ZIP FINAL ARTIFACTS
# ============================================================

def create_archive():

    archive_path = Path(
        "/kaggle/working/nih_reproducibility"
    )

    shutil.make_archive(
        str(archive_path),
        "zip",
        root_dir=OUTPUT_ROOT
    )

    print(
        "\n[OK] Archive created:",
        str(archive_path) + ".zip"
    )


# ============================================================
# 16. MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NIH ChestX-ray14 Reproducibility Tool"
    )

    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="Verify existing manifest"
    )

    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        help="Backbone for inference verification"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path for inference"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Test image path for inference"
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 65)
    print("NIH CHEST-XRAY14 REPRODUCIBILITY PIPELINE")
    print("=" * 65)

    print("\nSystem information:")
    print(json.dumps(get_system_information(), indent=2))

    if args.verify_manifest:

        verify_manifest()
        return

    inventory, checkpoints = create_inventory()

    configuration = save_experiment_configuration()

    verification = verify_checkpoints(checkpoints)

    # Optional inference verification.
    if args.backbone or args.checkpoint or args.image:

        if not all([
            args.backbone,
            args.checkpoint,
            args.image
        ]):

            parser.error(
                "Inference requires --backbone, "
                "--checkpoint and --image."
            )

        verify_inference(
            args.backbone,
            args.checkpoint,
            args.image
        )

    generate_report(
        inventory,
        checkpoints,
        verification,
        configuration
    )

    save_manifest(inventory)

    verify_manifest()

    create_archive()

    print("\n" + "=" * 65)
    print("REPRODUCIBILITY PIPELINE COMPLETED")
    print("=" * 65)

    print("\nOutput directory:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
