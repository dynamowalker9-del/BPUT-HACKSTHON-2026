

from pathlib import Path
from datetime import datetime
import os
import shutil

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset"
DERIVED = ROOT / "runs" / "finetune_data_nohelmet_nogloves"
INITIAL_WEIGHTS = ROOT / "runs/detect/runs/detect/yolo26n_ppe-5/weights/best.pt"
RUN_NAME = f"ppe_nohelmet_nogloves_finetune_{datetime.now():%Y%m%d_%H%M%S}"
CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots",
]
TARGET_IDS = {7, 9}


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_dataset() -> Path:
    if DERIVED.exists():
        yaml_path = DERIVED / "data.yaml"
        if yaml_path.is_file():
            return yaml_path
        raise FileExistsError(f"Derived dataset exists but is incomplete: {DERIVED}")

    duplicate_count = 0
    for split in ("train", "val", "test"):
        images_out = DERIVED / "images" / split
        labels_out = DERIVED / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image in (SOURCE / "images" / split).iterdir():
            if not image.is_file():
                continue
            label = SOURCE / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for {image}: {label}")

            link_file(image, images_out / image.name)
            link_file(label, labels_out / label.name)

            if split == "train":
                classes = set()
                for row in label.read_text(encoding="utf-8").splitlines():
                    fields = row.split()
                    if fields:
                        classes.add(int(fields[0]))
                if classes & TARGET_IDS:
                    duplicate_name = f"oversample_{image.name}"
                    link_file(image, images_out / duplicate_name)
                    link_file(label, labels_out / f"{Path(duplicate_name).stem}.txt")
                    duplicate_count += 1

    data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = DERIVED / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Derived train images added by oversampling: {duplicate_count}")
    return yaml_path


def main() -> None:
    if not INITIAL_WEIGHTS.is_file():
        raise FileNotFoundError(f"Fine-tuning checkpoint not found: {INITIAL_WEIGHTS}")
    data_yaml = build_dataset()

    model = YOLO(str(INITIAL_WEIGHTS))
    model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(ROOT / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=False,
        patience=20,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2,
        close_mosaic=10,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )
    print(f"BEST_CHECKPOINT {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
