
import argparse
import os
import sys
import yaml
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for PPE detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, help="Pretrained weights file (overrides config)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    parser.add_argument("--imgsz", type=int, help="Image size (overrides config)")
    parser.add_argument("--batch", type=int, help="Batch size (overrides config)")
    parser.add_argument("--device", type=str, help="Device: cpu or cuda index (overrides config)")
    parser.add_argument("--name", type=str, help="Experiment name (overrides config)")
    parser.add_argument("--project", type=str, help="Base project folder (overrides config)")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Determine final parameters (CLI overrides config)
    model_weights = args.model if args.model else cfg.get("model_weights", "yolo26n.pt")
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 150)
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640)
    batch = args.batch if args.batch is not None else cfg.get("batch", 16)
    device = args.device if args.device is not None else cfg.get("device", "cpu")
    experiment = args.name if args.name else cfg.get("experiment_name", "yolo26n_ppe")
    project = args.project if args.project else cfg.get("project_dir", "runs/detect")
    data_yaml = cfg.get("data_yaml", "dataset/dataset.yaml")

    # Validate files
    if not os.path.isfile(data_yaml):
        sys.exit(f"[ERROR] Dataset YAML not found: {data_yaml}")
    if not os.path.isfile(model_weights):
        sys.exit(f"[ERROR] Pretrained weights not found: {model_weights}")

    print("\n=== Training Configuration ===")
    print(f"Data YAML        : {data_yaml}")
    print(f"Pretrained weights: {model_weights}")
    print(f"Epochs           : {epochs}")
    print(f"Image size       : {imgsz}")
    print(f"Batch size       : {batch}")
    print(f"Device           : {device}")
    print(f"Experiment name  : {experiment}")
    print(f"Project folder   : {project}")
    print("===============================\n")

    # Load model
    model = YOLO(model_weights)

    # Build training args dict
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "conf": cfg.get("conf_thresh", 0.25),   # confidence filter for training
        "device": device,
        "name": experiment,
        "project": project,
        "save": True,
        "verbose": False,
        "patience": cfg.get("patience", 50),
        "plots": True,
    }

    print("[INFO] Launching training...")
    print("  Command (as if run on CLI): ultralytics train " +
          " ".join([f"{k}={v}" for k, v in train_args.items()]))
    print("\nTraining is now running. Check the console for progress logs.\n")

    # Train
    results = model.train(**train_args)

    # Report best checkpoint
    best_path = model.best
    if str(best_path).endswith(".pt"):
        print("\n✅ Training completed successfully!")
        print(f"Best checkpoint: {best_path}")
    else:
        print("\n❌ Training finished but no best.pt checkpoint was produced.")
        sys.exit(1)

if __name__ == "__main__":
    main()